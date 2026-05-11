"""
feature_engineering.py
=======================
Week 2 of the ML Pipeline — Feature Engineering, Model Training & SHAP

Dissertation: Fraud Detection in Credit Card Referral Systems
              Using Hybrid Rule-Based and Machine Learning Approach
BITS Pilani WILP — M.Tech Software Systems

═══════════════════════════════════════════════════════════════════════════════
WHAT IS THIS FILE DOING — READ THIS FIRST (especially if ML is new to you)
═══════════════════════════════════════════════════════════════════════════════

The synthetic dataset we generated has 35 raw columns — things like
referral_id, referrer_device_id, email_domain, timestamps, etc.

Machine learning models CANNOT work directly with:
  - String columns  (e.g. 'mailinator.com')
  - Boolean columns (True/False — needs to be 1/0)
  - Skewed numeric columns (e.g. min_gap_sec ranges from 10 to 86400 —
    a model treats 86400 as 860x more important than 100, which is wrong)
  - Raw columns that encode the same concept differently across rows

So this file does THREE things:

  STEP 1 — FEATURE ENGINEERING
    Convert raw columns into 16 numeric features that a model can learn from.
    E.g.  email_domain='mailinator.com'  →  is_disposable_email = 1
          min_gap_sec = 45               →  log_min_gap_sec = 3.83 (log scale)

  STEP 2 — TRAIN / TEST SPLIT + SMOTE
    Split the dataset into 80% training data and 20% test data.
    Apply SMOTE (oversampling) to the TRAINING set only to fix class imbalance
    (we have 82% legit vs 18% fraud — models learn the majority class and
    ignore minority class without this fix).

  STEP 3 — MODEL TRAINING AND EVALUATION
    Train two models:
      (a) Random Forest  — primary model (ensemble of decision trees)
      (b) Logistic Regression — baseline model (simple linear classifier)
    Evaluate both using Precision, Recall, F1-Score, AUC-ROC.
    Generate SHAP plots to explain what the model learned.

  STEP 4 — SAVE MODELS
    Save trained models to disk as .pkl files so the hybrid engine can
    load them without retraining.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS A MACHINE LEARNING MODEL — 30-SECOND EXPLANATION
═══════════════════════════════════════════════════════════════════════════════

Think of it like this:
  - You have 10,000 referral records, each with 16 numbers (features).
  - You also have the answer for each row: is_fraud = 0 or 1.
  - You show the model 8,000 of these rows (training set) and say:
    "Learn what combination of these 16 numbers predicts is_fraud=1."
  - The model finds patterns. E.g. it learns:
    "if rule_score > 0.7 AND device_collision = 1 → very likely fraud"
  - Then you test it on the 2,000 rows it has never seen (test set).
  - You measure how accurately it predicts fraud on those unseen rows.

Random Forest specifically works by:
  - Building 100 "decision trees", each trained on a slightly different
    random subset of the data.
  - Each tree votes: fraud or legit.
  - Final prediction = majority vote across all 100 trees.
  This makes it robust — no single tree is perfect, but 100 trees voting
  together are very accurate.

═══════════════════════════════════════════════════════════════════════════════
HOW TO RUN THIS FILE
═══════════════════════════════════════════════════════════════════════════════

Prerequisites — run these first:
  1.  python src/generate_dataset.py       → creates data/referral_dataset.csv
  2.  python src/rule_engine.py            → creates data/referral_dataset_scored.csv

Then run this file:
  python src/feature_engineering.py

Expected output:
  - Prints dataset shapes and class balance at each step
  - Prints classification report for both models
  - Prints AUC-ROC, Precision, Recall, F1 for both models
  - Saves:  data/train_features.csv
            data/test_features.csv
            models/random_forest.pkl
            models/logistic_regression.pkl
            models/scaler.pkl
            reports/figures/shap_summary.png
            reports/figures/shap_force_plot.png
            reports/figures/roc_curve.png
            reports/figures/confusion_matrix_rf.png
            reports/figures/feature_importance.png

═══════════════════════════════════════════════════════════════════════════════
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings('ignore')   # suppress sklearn version warnings

# ── Data handling ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Machine Learning — scikit-learn ──────────────────────────────────────────
# train_test_split : splits dataset into training set and test set
# StandardScaler   : normalises numeric features to mean=0, std=1
#                    (required for Logistic Regression; optional for RF)
# GridSearchCV     : tries different hyperparameter combinations automatically
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    classification_report,   # full precision/recall/F1 table
    confusion_matrix,         # 2x2 matrix: TP, FP, TN, FN
    roc_auc_score,            # Area Under the ROC Curve (0.5=random, 1.0=perfect)
    roc_curve,                # x=FPR, y=TPR at different thresholds
    precision_score,
    recall_score,
    f1_score,
)

# ── Class imbalance fix ───────────────────────────────────────────────────────
# SMOTE = Synthetic Minority Oversampling Technique
# Creates synthetic fraud records so the training set has a 50/50 split.
# CRITICAL: Apply ONLY to training data, NEVER to test data.
# Applying to test data would give falsely optimistic evaluation scores.
from imblearn.over_sampling import SMOTE

# ── Model persistence ─────────────────────────────────────────────────────────
# joblib saves a trained model object to a .pkl file so you can reload
# it later without retraining.
import joblib

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')               # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ── SHAP — model explainability ───────────────────────────────────────────────
# SHAP (SHapley Additive exPlanations) answers the question:
# "For THIS specific prediction, which features pushed the score toward
#  fraud and which pushed it toward legitimate?"
# It is critical for banking/compliance contexts where you must be able
# to explain WHY a credit-related decision was made.
import shap


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — edit these if you want to experiment
# ══════════════════════════════════════════════════════════════════════════════

# Domains that are classified as disposable / throwaway
DISPOSABLE_DOMAINS = {
    'mailinator.com', 'guerrillamail.com', 'throwaway.email',
    'temp-mail.org',  'fakeinbox.com',     'yopmail.com',
    'maildrop.cc',    'trashmail.com',     'sharklasers.com',
    'dispostable.com','getairmail.com',    'spamgourmet.com',
}

# These are the 16 features we will feed into the ML model.
# Everything else in the dataset is either an ID, a label, or raw text.
FEATURE_COLS = [
    'referral_velocity_7d',    # How many referrals this referrer sent in 7 days
    'referral_velocity_1hr',   # How many referrals in past hour (burst detection)
    'log_min_gap_sec',         # Log of minimum gap between referrals (seconds)
    'is_disposable_email',     # 1 if disposable email domain, else 0
    'ip_risk_score',           # 1 if referrer and referee share same IP, else 0
    'device_collision',        # 1 if same device both sides, else 0
    'account_age_ratio',       # referee_age / (referrer_age + 1)
    'identity_risk_score',     # Computed synthetic identity risk (0–1)
    'geo_risk_flag',           # 1 if geo_velocity_kmph > 500 (impossible travel)
    'cluster_risk_score',      # log1p(cluster_size) / 5 — ring participation score
    'promo_exploit_flag',      # 1 if dormant account activates right at promo start
    'address_collision_flag',  # 1 if same_address_count > 2
    'credit_age_mismatch',     # 1 if credit history older than account can explain
    'device_mutation_flag',    # 1 if device fingerprint in suspicious similarity range
    'rule_score',              # Normalised score from the rule engine (0–1)
    'rules_triggered_count',   # How many rules fired — cumulative signal count
]

# Label column — what we are trying to predict
TARGET_COL = 'is_fraud'

# Random seed — ensures experiments are reproducible
RANDOM_SEED = 42

# Paths
DATA_DIR    = 'data'
MODEL_DIR   = 'models'
FIG_DIR     = os.path.join('reports', 'figures')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw dataset columns into 16 numeric ML features.

    WHY THIS MATTERS:
      A Random Forest works by asking YES/NO questions about numbers.
      It cannot ask "is this email disposable?" — it can only ask
      "is is_disposable_email == 1?".
      Feature engineering is the translation layer between human domain
      knowledge and mathematical model inputs.

    Parameters
    ----------
    df : pd.DataFrame
        The scored dataset output by rule_engine.py.
        Must contain rule_score and rules_triggered columns.

    Returns
    -------
    pd.DataFrame
        Original df with 16 new feature columns appended.
    """
    df = df.copy()   # never modify the original dataframe in-place

    # ── Feature 1: referral_velocity_7d ─────────────────────────────────────
    # Direct copy — already a clean integer in the dataset.
    # High values indicate the referrer is submitting too many referrals.
    # Legitimate referrers rarely send more than 3 in 7 days.
    df['referral_velocity_7d'] = df['referrals_in_7d'].fillna(0).astype(int)

    # ── Feature 2: referral_velocity_1hr ────────────────────────────────────
    # Direct copy — same concept but over 1 hour window.
    # >3 in one hour is the burst velocity signal (Rule R09).
    df['referral_velocity_1hr'] = df['referrals_in_1hr'].fillna(0).astype(int)

    # ── Feature 3: log_min_gap_sec ───────────────────────────────────────────
    # WHY LOG TRANSFORM?
    # min_gap_sec ranges from 10 to 86400 (1 second to 1 day).
    # Without transformation, the model sees 86400 as 864x more important
    # than 100, which is not meaningful — both indicate "not burst".
    # np.log1p(x) = log(1 + x) which handles x=0 safely.
    # After transform: 10→2.40, 120→4.79, 3600→8.19, 86400→11.37
    # Now the difference between 10 and 120 (fraud range) is as visible
    # to the model as the difference between 3600 and 86400 (legit range).
    df['log_min_gap_sec'] = np.log1p(df['min_gap_sec'].fillna(0))

    # ── Feature 4: is_disposable_email ──────────────────────────────────────
    # Binary flag: 1 if the referee used a throwaway email domain.
    # Models cannot work with strings — we convert the domain to 0 or 1.
    df['is_disposable_email'] = (
        df['email_domain']
        .fillna('')
        .str.lower()
        .str.strip()
        .isin(DISPOSABLE_DOMAINS)
        .astype(int)
    )

    # ── Feature 5: ip_risk_score ─────────────────────────────────────────────
    # 1 if referrer and referee share the exact same IP address.
    # This is a weaker version of R01 (same device) — both being on the
    # same IP is suspicious but less conclusive than same device.
    df['ip_risk_score'] = (
        df['referrer_ip'].fillna('') == df['referee_ip'].fillna('')
    ).astype(int)

    # ── Feature 6: device_collision ─────────────────────────────────────────
    # 1 if referrer and referee used the same physical device.
    # This is the single strongest fraud signal in the dataset.
    # Encodes Rule R01 directly as a binary ML feature.
    df['device_collision'] = (
        df['referrer_device_id'].fillna('A') == df['referee_device_id'].fillna('B')
    ).astype(int)

    # ── Feature 7: account_age_ratio ────────────────────────────────────────
    # referee_age / (referrer_age + 1)
    # WHY THIS RATIO?
    # A legitimate referrer typically has an older account than the referee.
    # If a new account (referee_age=2) is being referred by another new
    # account (referrer_age=5), ratio = 2/6 = 0.33 — suspicious.
    # If a long-standing customer (referrer_age=1000) refers a new customer
    # (referee_age=30), ratio = 30/1001 = 0.03 — looks normal.
    # Adding 1 to denominator avoids division by zero.
    df['account_age_ratio'] = (
        df['referee_account_age_days'].fillna(0)
        / (df['referrer_account_age_days'].fillna(1) + 1)
    ).round(4)

    # ── Feature 8: identity_risk_score ──────────────────────────────────────
    # Direct copy — already a float between 0 and 1.
    # Encodes synthetic identity risk. High values (>0.7) trigger Rule R08.
    # The ML model can learn more nuanced thresholds than the rule engine's
    # hard 0.7 cutoff.
    df['identity_risk_score'] = df['identity_risk_score'].fillna(0.0)

    # ── Feature 9: geo_risk_flag ─────────────────────────────────────────────
    # 1 if geo_velocity_kmph > 500 (physically very difficult travel speed).
    # We use 500 here (softer than R10's hard 900) so the ML model can learn
    # from borderline cases too, not just the extreme ones the rule catches.
    df['geo_risk_flag'] = (df['geo_velocity_kmph'].fillna(0) > 500).astype(int)

    # ── Feature 10: cluster_risk_score ──────────────────────────────────────
    # log1p(cluster_size) / 5
    # WHY NOT JUST cluster_size?
    # cluster_size of 4 vs 5 is equally suspicious — both are rings.
    # But cluster_size of 1 (no ring) vs 4 (ring) is very meaningful.
    # Log transform compresses the high end and amplifies the 1→4 jump.
    # Dividing by 5 keeps the feature in roughly 0–1 range:
    #   size=1 → 0.14,  size=4 → 0.34,  size=8 → 0.44
    df['cluster_risk_score'] = (
        np.log1p(df['referral_cluster_size'].fillna(1)) / 5.0
    ).round(4)

    # ── Feature 11: promo_exploit_flag ──────────────────────────────────────
    # 1 if ALL THREE conditions are met simultaneously:
    #   - A promotion is currently active
    #   - The referral was submitted within 10 minutes of promo start
    #   - The referrer was dormant for > 90 days before this
    # A legitimate user might submit during a promo (common), but a dormant
    # account activating WITHIN MINUTES of a promo starting is suspicious.
    df['promo_exploit_flag'] = (
        (df['promo_active'].fillna(False).astype(bool)) &
        (df['promo_start_gap_min'].fillna(9999).between(0, 9)) &
        (df['referrer_dormancy_days'].fillna(0) > 90)
    ).astype(int)

    # ── Feature 12: address_collision_flag ──────────────────────────────────
    # 1 if more than 2 accounts are registered at the same address in 30 days.
    # Encodes Rule R11. The ML model can learn to weight this differently
    # depending on what other features are also present.
    df['address_collision_flag'] = (
        df['same_address_count'].fillna(1) > 2
    ).astype(int)

    # ── Feature 13: credit_age_mismatch ─────────────────────────────────────
    # 1 if the referee claims a credit history > 60 months old
    #   BUT their account is less than 30 days old.
    # This is physically impossible for a genuine person — you cannot have
    # a 5-year credit history if your account was created 3 weeks ago.
    # Encodes Rule R17.
    df['credit_age_mismatch'] = (
        (df['credit_age_months'].fillna(0)         > 60) &
        (df['referee_account_age_days'].fillna(9999) < 30)
    ).astype(int)

    # ── Feature 14: device_mutation_flag ────────────────────────────────────
    # 1 if device_similarity_score is between 0.75 and 0.99.
    # Score of 1.0 = exact same device (caught by R01/device_collision).
    # Score < 0.75 = genuinely different device.
    # Score 0.75–0.99 = suspiciously similar but not identical — indicates
    # deliberate minor modification of browser fingerprint to evade detection.
    df['device_mutation_flag'] = (
        df['device_similarity_score'].fillna(0.0).between(0.75, 0.99)
    ).astype(int)

    # ── Feature 15: rule_score ───────────────────────────────────────────────
    # The normalised score (0–1) output by the Rule Engine.
    # WHY USE RULE OUTPUT AS AN ML FEATURE?
    # The rule engine encodes expert domain knowledge about fraud patterns.
    # By feeding rule_score into the ML model, we allow the model to learn
    # "when the rules say something is risky, how much extra weight should
    # I give to that compared to just the raw features?"
    # This is a key part of what makes the hybrid approach work.
    # If rule_score column is missing (rule engine not run), default to 0.
    if 'rule_score' in df.columns:
        df['rule_score'] = df['rule_score'].fillna(0.0)
    else:
        print("  WARNING: rule_score column not found. "
              "Run rule_engine.py first for best results. Defaulting to 0.")
        df['rule_score'] = 0.0

    # ── Feature 16: rules_triggered_count ───────────────────────────────────
    # How many individual rules fired for this record.
    # A record that triggers 4 rules is more suspicious than one that
    # triggers 1 rule, even if both have similar rule_score.
    # This feature captures the NUMBER of independent signals present.
    if 'rules_triggered' in df.columns:
        df['rules_triggered_count'] = df['rules_triggered'].apply(
            lambda x: 0 if (pd.isna(x) or str(x) == 'NONE' or str(x) == '[]')
                      else len(str(x).replace('[','').replace(']','')
                                    .replace("'",'').split(','))
        )
    else:
        df['rules_triggered_count'] = 0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT + SMOTE
# ══════════════════════════════════════════════════════════════════════════════

def split_and_resample(df: pd.DataFrame):
    """
    Split the feature-engineered dataset into training and test sets,
    then apply SMOTE to the training set to fix class imbalance.

    WHY DO WE SPLIT BEFORE SMOTE?
    ─────────────────────────────
    This is one of the most common mistakes beginners make in ML.

    WRONG approach (data leakage):
      1. Apply SMOTE to full dataset   ← WRONG — test rows leak into training
      2. Split into train/test

    CORRECT approach:
      1. Split into train/test FIRST   ← test set is sealed away
      2. Apply SMOTE ONLY to train set ← test set stays pure

    If you apply SMOTE before splitting, the synthetic fraud rows it creates
    are derived from test-set rows — the model effectively "sees" the test
    set during training, making evaluation scores artificially high.

    WHAT IS SMOTE?
    ──────────────
    Our dataset: 82% legitimate, 18% fraud.
    Without correction, a model that ALWAYS predicts "legitimate" would score
    82% accuracy — but it would detect ZERO fraud. This is useless.

    SMOTE fixes this by creating synthetic fraud records in the training set.
    It finds existing fraud records and creates new ones that are "in between"
    real fraud records in feature space.
    After SMOTE: training set is ~50% legitimate, ~50% fraud.

    The test set is NEVER touched — it stays at 82%/18% to reflect real
    world conditions, so your evaluation metrics are honest.

    WHAT IS STRATIFIED SPLIT?
    ──────────────────────────
    stratify=y ensures that BOTH the train and test sets have the same
    fraud ratio (18%). Without this, you might get unlucky and have all
    fraud records in training and none in test (or vice versa).

    Parameters
    ----------
    df : pd.DataFrame with FEATURE_COLS and TARGET_COL present

    Returns
    -------
    X_train_res, X_test, y_train_res, y_test, scaler
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print(f"\n{'─'*60}")
    print("STEP 2: Train/Test Split + SMOTE")
    print(f"{'─'*60}")
    print(f"  Total records        : {len(df):,}")
    print(f"  Total fraud records  : {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"  Total legit records  : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")

    # ── 2a. Stratified 80/20 split ────────────────────────────────────────────
    # test_size=0.2  → 20% of data held out for testing
    # stratify=y     → preserve 82%/18% ratio in both splits
    # random_state   → reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        stratify     = y,
        random_state = RANDOM_SEED,
    )

    print(f"\n  After 80/20 split:")
    print(f"    Train: {len(X_train):,} records  "
          f"(fraud: {y_train.sum():,} = {y_train.mean()*100:.1f}%)")
    print(f"    Test : {len(X_test):,} records  "
          f"(fraud: {y_test.sum():,} = {y_test.mean()*100:.1f}%)")

    # ── 2b. Feature scaling for Logistic Regression ───────────────────────────
    # WHY SCALE?
    # Logistic Regression is sensitive to feature magnitudes.
    # If one feature ranges 0–1 and another ranges 0–10000, the model
    # gives disproportionate weight to the larger one.
    # StandardScaler converts every feature to: (value - mean) / std_dev
    # Result: every feature has mean=0 and std=1 → same scale.
    # Random Forest does NOT need scaling (it uses thresholds, not distances),
    # but we scale anyway so both models use the same features.
    # CRITICAL: fit the scaler on TRAINING data only.
    # Then transform BOTH train and test using the training statistics.
    # If you fit on the full dataset, test statistics leak into training.
    scaler            = StandardScaler()
    X_train_scaled    = scaler.fit_transform(X_train)    # fit + transform train
    X_test_scaled     = scaler.transform(X_test)         # transform only (no fit)

    # Convert back to DataFrames with feature names (easier to debug)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS)

    # ── 2c. SMOTE on training set only ────────────────────────────────────────
    # k_neighbors=5 means: to create each synthetic fraud record, find the
    # 5 nearest existing fraud records and interpolate between them.
    smote             = SMOTE(k_neighbors=5, random_state=RANDOM_SEED)

    # We apply SMOTE on the SCALED training features
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled_df, y_train)

    print(f"\n  After SMOTE (training set only):")
    print(f"    Train: {len(X_train_res):,} records  "
          f"(fraud: {y_train_res.sum():,} = {y_train_res.mean()*100:.1f}%)")
    print(f"    Test unchanged: {len(X_test):,} records  "
          f"(fraud: {y_test.sum():,} = {y_test.mean()*100:.1f}%)")
    print(f"    New synthetic fraud records created: "
          f"{len(X_train_res) - len(X_train_scaled_df):,}")

    return X_train_res, X_test_scaled_df, y_train_res, y_test, scaler, X_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a — RANDOM FOREST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train the primary model: Random Forest Classifier.

    WHAT IS A RANDOM FOREST?
    ─────────────────────────
    A Random Forest is an "ensemble" of decision trees.
    Each tree asks a series of YES/NO questions about the features.
    E.g.:
      "Is rule_score > 0.5?"
        YES → "Is device_collision == 1?"
                YES → Predict FRAUD
                NO  → Predict LEGITIMATE
        NO  → Predict LEGITIMATE

    One tree is weak. But 100 trees voting together are strong —
    each tree was trained on a different random subset of the data,
    so they make different mistakes. When they vote together, mistakes
    cancel out.

    KEY PARAMETERS:
      n_estimators=100  → build 100 trees
      max_depth=10      → each tree can ask at most 10 questions deep
                          (prevents overfitting — trees that are too
                          deep memorise training data instead of learning)
      class_weight='balanced' → automatically weight fraud class higher
                          to compensate for imbalance (in addition to SMOTE)
      n_jobs=-1         → use all CPU cores (faster training)

    WHAT IS OVERFITTING?
    ─────────────────────
    Overfitting = the model memorises training data instead of learning
    general patterns. It scores very high on training data but poorly
    on test data (which it has never seen).
    max_depth=10 and min_samples_leaf=5 limit tree depth/size to prevent
    this.

    Parameters
    ----------
    X_train, y_train : training features and labels (after SMOTE)
    X_test,  y_test  : test features and labels (untouched)

    Returns
    -------
    rf_model : trained RandomForestClassifier
    rf_probs : array of fraud probabilities for each test record (0–1)
    """
    print(f"\n{'─'*60}")
    print("STEP 3a: Training Random Forest Classifier")
    print(f"{'─'*60}")

    rf_model = RandomForestClassifier(
        n_estimators      = 100,     # number of trees
        max_depth         = 10,      # max depth per tree
        min_samples_leaf  = 5,       # each leaf must have at least 5 samples
        class_weight      = 'balanced',
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )

    print("  Training Random Forest on SMOTE-resampled training set...")
    rf_model.fit(X_train, y_train)
    print("  Training complete.")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    # predict()       → hard label: 0 or 1
    # predict_proba() → probability of each class: [P(legit), P(fraud)]
    # We want [:, 1] — the probability of being fraud (second column)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    print(f"\n  Random Forest — Test Set Evaluation:")
    print(f"  {'─'*50}")

    # UNDERSTANDING THE METRICS:
    # ─────────────────────────────────────────────────────────────────
    # Precision = of all records we PREDICTED as fraud,
    #             what fraction was ACTUALLY fraud?
    #             High precision = few false alarms (legit blocked)
    #
    # Recall    = of all records that ARE actually fraud,
    #             what fraction did we CATCH?
    #             High recall = few frauds missed
    #
    # F1-Score  = harmonic mean of precision and recall.
    #             Best single metric when you care about BOTH.
    #
    # AUC-ROC   = Area Under the ROC Curve.
    #             Measures how well the model RANKS fraud higher than legit
    #             regardless of the threshold you choose.
    #             0.5 = random guessing, 1.0 = perfect model.
    #
    # In fraud detection for banking:
    #   High RECALL is usually more important than high precision.
    #   Missing a fraud costs more than blocking a legitimate user.
    # ─────────────────────────────────────────────────────────────────

    print(classification_report(
        y_test, rf_preds,
        target_names=['Legitimate', 'Fraud'],
        digits=4,
    ))

    auc = roc_auc_score(y_test, rf_probs)
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Precision: {precision_score(y_test, rf_preds):.4f}")
    print(f"  Recall   : {recall_score(y_test, rf_preds):.4f}")
    print(f"  F1-Score : {f1_score(y_test, rf_preds):.4f}")

    return rf_model, rf_probs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b — LOGISTIC REGRESSION (BASELINE)
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train the baseline model: Logistic Regression.

    WHAT IS LOGISTIC REGRESSION?
    ─────────────────────────────
    Despite the name, Logistic Regression is a CLASSIFICATION algorithm.
    It learns a linear boundary that separates fraud from legitimate.
    It computes:
      P(fraud) = sigmoid( w1*feature1 + w2*feature2 + ... + b )
    where w1, w2... are weights learned from training data.

    WHY USE IT AS A BASELINE?
    ──────────────────────────
    1. It is the simplest possible classifier.
    2. It is highly interpretable — you can see the weight it assigns
       to each feature and understand the model directly.
    3. It serves as a benchmark: if Random Forest does not significantly
       beat Logistic Regression, the data or features may be too simple
       and you need to rethink the approach.
    4. In your dissertation's evaluation chapter, showing that RF beats
       LR demonstrates the value of using a more complex model.

    Parameters
    ----------
    X_train, y_train : training features (SCALED) and labels (after SMOTE)
    X_test,  y_test  : test features (SCALED) and labels

    Returns
    -------
    lr_model : trained LogisticRegression
    lr_probs : array of fraud probabilities for each test record
    """
    print(f"\n{'─'*60}")
    print("STEP 3b: Training Logistic Regression (Baseline)")
    print(f"{'─'*60}")

    lr_model = LogisticRegression(
        max_iter     = 1000,          # max iterations for convergence
        class_weight = 'balanced',   # compensate for imbalance
        solver       = 'lbfgs',      # optimisation algorithm
        random_state = RANDOM_SEED,
        C            = 1.0,          # regularisation strength (default)
    )

    print("  Training Logistic Regression on SMOTE-resampled training set...")
    lr_model.fit(X_train, y_train)
    print("  Training complete.")

    lr_preds = lr_model.predict(X_test)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    print(f"\n  Logistic Regression — Test Set Evaluation:")
    print(f"  {'─'*50}")
    print(classification_report(
        y_test, lr_preds,
        target_names=['Legitimate', 'Fraud'],
        digits=4,
    ))
    auc = roc_auc_score(y_test, lr_probs)
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Precision: {precision_score(y_test, lr_preds):.4f}")
    print(f"  Recall   : {recall_score(y_test, lr_preds):.4f}")
    print(f"  F1-Score : {f1_score(y_test, lr_preds):.4f}")

    # ── Feature coefficients — unique to LR, not available in RF ─────────────
    coef_df = pd.DataFrame({
        'feature':     FEATURE_COLS,
        'coefficient': lr_model.coef_[0],
    }).sort_values('coefficient', ascending=False)
    print(f"\n  Logistic Regression — Feature Coefficients:")
    print(f"  (Positive = pushes toward FRAUD, Negative = pushes toward LEGIT)")
    print(coef_df.to_string(index=False))

    return lr_model, lr_probs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def generate_shap_plots(rf_model, X_test: pd.DataFrame, fig_dir: str):
    """
    Generate SHAP explainability plots for the Random Forest model.

    WHAT IS SHAP AND WHY DOES IT MATTER FOR BANKING?
    ──────────────────────────────────────────────────
    SHAP = SHapley Additive exPlanations (from game theory)

    The problem with ML models: they produce a score (e.g., 0.82 fraud
    probability) but don't say WHY. This is unacceptable in banking
    because:
      - RBI/regulatory guidelines require explainability for credit decisions
      - Compliance teams need to know why a referral was blocked
      - Fraud analysts need to understand what the model learned

    SHAP solves this by computing, for each prediction:
    "How much did each feature CONTRIBUTE to pushing this score
     above or below the average?"

    Example output for one fraud record:
      rule_score         : +0.42  (pushed strongly toward fraud)
      device_collision   : +0.31  (pushed toward fraud)
      identity_risk_score: +0.18  (pushed toward fraud)
      log_min_gap_sec    : -0.05  (pushed slightly toward legit)
      is_disposable_email: +0.08  (pushed toward fraud)

    This tells the compliance officer: "This referral was blocked mainly
    because the rule engine flagged it AND the same device was used."

    PLOTS GENERATED:
    ─────────────────
    1. shap_summary.png  — Shows all 16 features ranked by their average
                           impact on fraud predictions across ALL test records.
                           Dots are individual records, color = feature value.
                           Use this in dissertation Chapter 5.

    2. shap_force_plot.png — For ONE specific fraud record, shows exactly
                           which features pushed the score up/down.
                           Use this as an example in the discussion chapter.

    Parameters
    ----------
    rf_model : trained RandomForestClassifier
    X_test   : test features DataFrame (unscaled — RF doesn't need scaling)
    fig_dir  : output directory for saving plots
    """
    print(f"\n{'─'*60}")
    print("STEP 4: Generating SHAP Explainability Plots")
    print(f"{'─'*60}")
    print("  Computing SHAP values (this may take 1–2 minutes)...")

    # TreeExplainer is the fast SHAP variant for tree-based models like RF
    explainer   = shap.TreeExplainer(rf_model)

    # Compute SHAP values for ALL test records
    # shap_values is a list: [values_for_class_0, values_for_class_1]
    # We want class 1 (fraud) → shap_values[1]
    # Shape: (n_test_records, 16_features)
    shap_values = explainer.shap_values(X_test)

    # Handle both old and new shap API return formats
    if isinstance(shap_values, list):
        sv_fraud = shap_values[1]   # class 1 = fraud
    else:
        sv_fraud = shap_values

    print("  SHAP values computed.")

    # ── Plot 1: Summary plot (beeswarm) ──────────────────────────────────────
    # This is the most important SHAP plot.
    # Y-axis: features ranked by mean |SHAP value| (most important at top)
    # X-axis: SHAP value (positive = pushes toward fraud)
    # Each dot = one test record.
    # Color: red = high feature value, blue = low feature value.
    print("  Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_fraud,
        X_test,
        feature_names = FEATURE_COLS,
        show          = False,
        plot_size     = None,
    )
    plt.title('SHAP Feature Importance — Random Forest (Fraud Class)',
              fontsize=13, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(fig_dir, 'shap_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {summary_path}")

    # ── Plot 2: Force plot for a single high-risk record ─────────────────────
    # Find a test record with high fraud probability to use as the example.
    # We use the record with the highest sum of positive SHAP values.
    print("  Generating SHAP force plot (single record example)...")
    row_idx = int(np.argmax(sv_fraud.sum(axis=1)))

    # matplotlib=True saves as image (the default HTML version needs a browser)
    shap.force_plot(
        base_value     = explainer.expected_value[1]
                         if isinstance(explainer.expected_value, list)
                         else explainer.expected_value,
        shap_values    = sv_fraud[row_idx],
        features       = X_test.iloc[row_idx],
        feature_names  = FEATURE_COLS,
        matplotlib     = True,
        show           = False,
    )
    plt.title(f'SHAP Force Plot — Test Record #{row_idx} (High-Risk Example)',
              fontsize=11)
    plt.tight_layout()
    force_path = os.path.join(fig_dir, 'shap_force_plot.png')
    plt.savefig(force_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {force_path}")

    # ── Print top 5 features for the example record ───────────────────────────
    print(f"\n  Top SHAP contributors for example record #{row_idx}:")
    contributions = sorted(
        zip(FEATURE_COLS, sv_fraud[row_idx]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feat, val in contributions[:5]:
        direction = "→ FRAUD" if val > 0 else "→ LEGIT"
        print(f"    {feat:<30}  SHAP={val:+.4f}  {direction}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISATIONS (Confusion Matrix, ROC Curve, Feature Importance)
# ══════════════════════════════════════════════════════════════════════════════

def generate_evaluation_plots(rf_model, lr_model,
                               X_test_scaled, y_test,
                               rf_probs, lr_probs,
                               X_test_unscaled,
                               fig_dir: str):
    """
    Generate all evaluation visualisations needed for the dissertation.

    PLOTS:
    ──────
    1. confusion_matrix_rf.png — 2x2 matrix showing TP, TN, FP, FN for RF
    2. roc_curve.png           — ROC curves for RF, LR, and Rule Engine
    3. feature_importance.png  — RF built-in feature importances (Gini)

    UNDERSTANDING THE CONFUSION MATRIX:
    ──────────────────────────────────────
    For a fraud detection system:

                        PREDICTED
                    Legit    Fraud
    ACTUAL  Legit  [  TN  |  FP  ]   FP = False Positive = Legit blocked
            Fraud  [  FN  |  TP  ]   FN = False Negative = Fraud missed

    In banking context:
      FN (missed fraud) is very costly — real fraud gets through
      FP (false alarm)  is also costly — legitimate users are blocked

    For fraud detection, we want:
      High TP (catching fraud)  → controlled by Recall
      Low  FP (not over-blocking) → controlled by Precision

    The Hybrid approach in Week 3 will aim to tune these thresholds.
    """
    print(f"\n{'─'*60}")
    print("STEP 5: Generating Evaluation Visualisations")
    print(f"{'─'*60}")

    # ── Plot 1: Confusion Matrix for Random Forest ────────────────────────────
    rf_preds = rf_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, rf_preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Predicted Legit', 'Predicted Fraud'],
        yticklabels=['Actual Legit',    'Actual Fraud'],
    )
    ax.set_title('Confusion Matrix — Random Forest', fontsize=13, pad=12)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)

    # Annotate the cells with labels
    tn, fp, fn, tp = cm.ravel()
    ax.texts[0].set_text(f'TN\n{tn:,}')
    ax.texts[1].set_text(f'FP\n{fp:,}')
    ax.texts[2].set_text(f'FN\n{fn:,}')
    ax.texts[3].set_text(f'TP\n{tp:,}')

    plt.tight_layout()
    cm_path = os.path.join(fig_dir, 'confusion_matrix_rf.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cm_path}")
    print(f"  Confusion Matrix: TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    # ── Plot 2: ROC Curve — RF vs LR ─────────────────────────────────────────
    # ROC Curve plots:
    #   X-axis: False Positive Rate (FPR) = FP / (FP + TN)
    #   Y-axis: True Positive Rate  (TPR) = TP / (TP + FN)  = Recall
    # At each decision threshold (0.0 to 1.0), you get one (FPR, TPR) point.
    # The curve shows the trade-off: to catch more fraud (higher TPR),
    # you also block more legitimate users (higher FPR).
    # AUC = area under this curve. 0.5 = random, 1.0 = perfect.
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    auc_rf = roc_auc_score(y_test, rf_probs)
    auc_lr = roc_auc_score(y_test, lr_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_rf, tpr_rf, 'b-',  lw=2,
            label=f'Random Forest   (AUC = {auc_rf:.4f})')
    ax.plot(fpr_lr, tpr_lr, 'g--', lw=2,
            label=f'Logistic Regr.  (AUC = {auc_lr:.4f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=1, label='Random Baseline (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_title('ROC Curve — Random Forest vs Logistic Regression', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {roc_path}")

    # ── Plot 3: RF Feature Importances (Gini / Mean Decrease in Impurity) ────
    # Random Forest computes a built-in feature importance based on how much
    # each feature reduces "impurity" (uncertainty) across all tree splits.
    # This is separate from SHAP — it tells you which features the model
    # used most for splitting decisions, across ALL predictions.
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': importances,
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(fi_df['feature'], fi_df['importance'],
                   color='steelblue', edgecolor='white')
    ax.set_xlabel('Gini Importance (Mean Decrease in Impurity)', fontsize=11)
    ax.set_title('Random Forest — Feature Importances', fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, fi_df['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', fontsize=9)
    plt.tight_layout()
    fi_path = os.path.join(fig_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fi_path}")
    print(f"\n  Top 5 features by Gini importance:")
    for _, row in fi_df.tail(5).iterrows():
        print(f"    {row['feature']:<30}  {row['importance']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def save_models(rf_model, lr_model, scaler, model_dir: str):
    """
    Save trained models to disk as .pkl files.

    WHY SAVE MODELS?
    ─────────────────
    Training takes time. Once trained, you save the model to disk.
    Later, the hybrid engine just loads the saved model with joblib.load()
    and uses it immediately — no retraining needed.

    This is exactly how production ML systems work: train offline,
    deploy the saved model file for real-time scoring.

    Files saved:
      random_forest.pkl       — primary fraud detection model
      logistic_regression.pkl — baseline model (for comparison)
      scaler.pkl              — the StandardScaler fitted on training data
                                MUST be saved to apply the same transformation
                                to new data at prediction time
    """
    os.makedirs(model_dir, exist_ok=True)
    rf_path  = os.path.join(model_dir, 'random_forest.pkl')
    lr_path  = os.path.join(model_dir, 'logistic_regression.pkl')
    sc_path  = os.path.join(model_dir, 'scaler.pkl')

    joblib.dump(rf_model,  rf_path)
    joblib.dump(lr_model,  lr_path)
    joblib.dump(scaler,    sc_path)

    print(f"\n{'─'*60}")
    print("STEP 6: Models saved")
    print(f"{'─'*60}")
    print(f"  Random Forest        → {rf_path}")
    print(f"  Logistic Regression  → {lr_path}")
    print(f"  Scaler               → {sc_path}")
    print(f"\n  Load them later with:")
    print(f"    import joblib")
    print(f"    rf = joblib.load('{rf_path}')")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — orchestrates all steps in order
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,   exist_ok=True)

    print("="*60)
    print("WEEK 2 — ML PIPELINE: Feature Engineering & Model Training")
    print("="*60)

    # ── Load the scored dataset (output of rule_engine.py) ───────────────────
    scored_path = os.path.join(DATA_DIR, 'referral_dataset_scored.csv')
    raw_path    = os.path.join(DATA_DIR, 'referral_dataset.csv')

    if os.path.exists(scored_path):
        print(f"\nLoading scored dataset from: {scored_path}")
        df = pd.read_csv(scored_path)
    elif os.path.exists(raw_path):
        print(f"\nWARNING: Scored dataset not found. Loading raw dataset: {raw_path}")
        print("         Rule engine features (rule_score, rules_triggered) will be 0.")
        df = pd.read_csv(raw_path)
    else:
        print(f"ERROR: No dataset found.")
        print(f"  Run:  python src/generate_dataset.py   first")
        print(f"  Then: python src/rule_engine.py")
        raise SystemExit(1)

    print(f"  Loaded {len(df):,} records, {df.shape[1]} columns")

    # ── STEP 1: Feature Engineering ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STEP 1: Feature Engineering")
    print(f"{'─'*60}")
    print(f"  Engineering {len(FEATURE_COLS)} features from raw columns...")
    df_feat = engineer_features(df)

    # Quick sanity check — no NaN in feature columns after engineering
    null_counts = df_feat[FEATURE_COLS].isnull().sum()
    if null_counts.any():
        print(f"  WARNING: NaN values found after feature engineering:")
        print(null_counts[null_counts > 0])
    else:
        print(f"  All {len(FEATURE_COLS)} features engineered — no missing values.")

    # Save feature-engineered data for reference
    feat_path = os.path.join(DATA_DIR, 'referral_dataset_features.csv')
    df_feat[['referral_id', TARGET_COL, 'fraud_type'] + FEATURE_COLS].to_csv(
        feat_path, index=False
    )
    print(f"  Feature dataset saved to: {feat_path}")

    # ── STEP 2: Train/Test Split + SMOTE ─────────────────────────────────────
    (X_train_res, X_test_scaled,
     y_train_res, y_test,
     scaler, X_test_unscaled) = split_and_resample(df_feat)

    # Save train/test splits (useful for reproducing results)
    train_df = pd.DataFrame(X_train_res, columns=FEATURE_COLS)
    train_df[TARGET_COL] = y_train_res.values
    train_df.to_csv(os.path.join(DATA_DIR, 'train_features.csv'), index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS)
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(os.path.join(DATA_DIR, 'test_features.csv'), index=False)
    print(f"\n  Train/test CSVs saved to data/")

    # ── STEP 3a: Train Random Forest ─────────────────────────────────────────
    rf_model, rf_probs = train_random_forest(
        X_train_res, y_train_res, X_test_scaled, y_test
    )

    # ── STEP 3b: Train Logistic Regression ───────────────────────────────────
    lr_model, lr_probs = train_logistic_regression(
        X_train_res, y_train_res, X_test_scaled, y_test
    )

    # ── STEP 4: SHAP Explainability ───────────────────────────────────────────
    # Note: SHAP TreeExplainer works with unscaled features for RF
    # (RF doesn't need scaling — we use the unscaled X_test here so that
    # the SHAP feature values in the plots show the original feature ranges)
    X_test_for_shap = pd.DataFrame(X_test_unscaled, columns=FEATURE_COLS)
    generate_shap_plots(rf_model, X_test_for_shap, FIG_DIR)

    # ── STEP 5: Evaluation Plots ──────────────────────────────────────────────
    generate_evaluation_plots(
        rf_model, lr_model,
        X_test_scaled, y_test,
        rf_probs, lr_probs,
        X_test_for_shap,
        FIG_DIR,
    )

    # ── STEP 6: Save Models ───────────────────────────────────────────────────
    save_models(rf_model, lr_model, scaler, MODEL_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("WEEK 2 COMPLETE — Summary")
    print(f"{'='*60}")
    print(f"  Engineered features  : {len(FEATURE_COLS)}")
    print(f"  Training records     : {len(X_train_res):,} (after SMOTE)")
    print(f"  Test records         : {len(X_test_scaled):,}")
    print(f"  Models trained       : Random Forest, Logistic Regression")
    print(f"  Models saved to      : {MODEL_DIR}/")
    print(f"  Plots saved to       : {FIG_DIR}/")
    print(f"\n  Next step: Run  python src/hybrid_engine.py")
    print(f"  That file will combine rule_score + ml_probability")
    print(f"  into a final APPROVE / REVIEW / BLOCK decision.")
    print("="*60)


if __name__ == '__main__':
    main()
