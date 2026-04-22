"""
rule_engine.py
==============
Rule-Based Fraud Detection Engine
Dissertation: Fraud Detection in Credit Card Referral Systems
              Using Hybrid Rule-Based and Machine Learning Approach
BITS Pilani WILP — M.Tech Software Systems

Design principles (from Section 4.1 of mid-sem report):
  1. Modularity   — each rule is an independent method; add/disable without
                    touching the rest of the engine.
  2. Auditability — every triggered rule is recorded by ID in the output,
                    giving a human-readable explanation of the risk score.
  3. Determinism  — same input always produces same output; required for
                    compliance in credit decision systems.

Rule score formula (Section 4.3):
  rule_score = min( sum(triggered_weights) / 100.0, 1.0 )

Hard-block rules (Section 4.2):
  R01, R05, R10, R13, R14 — these bypass the ML model entirely.

Dataset-wide lookup rules (R04, R13, R15):
  These rules cannot evaluate a single row in isolation — they need to
  know counts across the full dataset.  Call load_dataset(df) once after
  reading the CSV; the engine then builds internal lookup dictionaries
  that make per-row evaluation O(1).

Usage — single record:
    engine = RuleEngine()
    engine.load_dataset(df)          # build lookups — call once
    result = engine.evaluate(row)    # row is a dict or pd.Series

Usage — full dataframe (batch):
    engine = RuleEngine()
    engine.load_dataset(df)
    results = df.apply(lambda r: engine.evaluate(r.to_dict()), axis=1)
    df['rule_score']      = results.apply(lambda x: x['rule_score'])
    df['rules_triggered'] = results.apply(lambda x: x['rules_triggered'])
    df['is_hard_block']   = results.apply(lambda x: x['is_hard_block'])
"""

from __future__ import annotations
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

# Hard-block rule IDs — any one of these triggers an immediate BLOCK
# regardless of the ML score (Section 4.2)
HARD_BLOCK_RULES = {'R01', 'R05', 'R10', 'R13', 'R14'}

# Disposable / throwaway email domains (R03)
DISPOSABLE_DOMAINS = {
    'mailinator.com', 'guerrillamail.com', 'throwaway.email',
    'temp-mail.org',  'fakeinbox.com',     'yopmail.com',
    'maildrop.cc',    'trashmail.com',     'sharklasers.com',
    'dispostable.com','getairmail.com',    'spamgourmet.com',
}

# Thresholds — keep them in one place so they are easy to tune
T_SAME_DEVICE_WEIGHT        = 95    # R01
T_HIGH_VELOCITY_7D          = 5     # R02  referrals_in_7d  > this
T_HIGH_VELOCITY_WEIGHT      = 75    # R02
T_DISPOSABLE_EMAIL_WEIGHT   = 70    # R03
T_IP_REUSE_COUNT            = 3     # R04  ip_account_count > this
T_IP_REUSE_WEIGHT           = 80    # R04
T_RING_CLUSTER_SIZE         = 3     # R05  referral_cluster_size > this
T_RING_WEIGHT               = 90    # R05
T_GEO_MISMATCH_WEIGHT       = 60    # R06
T_NEW_ACCOUNT_AGE_DAYS      = 3     # R07  referrer_account_age_days < this
T_NEW_ACCOUNT_REFERRALS     = 2     # R07  referrals_in_7d > this
T_NEW_ACCOUNT_WEIGHT        = 65    # R07
T_IDENTITY_RISK_SCORE       = 0.7   # R08  identity_risk_score > this
T_IDENTITY_RISK_WEIGHT      = 70    # R08
T_BURST_1HR                 = 3     # R09  referrals_in_1hr > this
T_BURST_GAP_SEC             = 120   # R09  min_gap_sec < this
T_BURST_WEIGHT              = 80    # R09
T_GEO_VELOCITY_KMPH         = 900   # R10  geo_velocity_kmph > this (hard block)
T_GEO_VELOCITY_WEIGHT       = 85    # R10
T_ADDRESS_COUNT             = 2     # R11  same_address_count > this
T_ADDRESS_WEIGHT            = 55    # R11
T_CASH_ADVANCE_RATIO        = 0.8   # R12  cash_advance_ratio > this
T_NO_USAGE_WEIGHT           = 65    # R12
T_PAN_REUSE_COUNT           = 1     # R13  pan seen in > this many records (hard block)
T_PAN_REUSE_WEIGHT          = 95    # R13
T_MULE_ACCOUNT_AGE          = 7     # R14  referee_account_age_days < this
T_MULE_CLUSTER_SIZE         = 3     # R14  referral_cluster_size > this (hard block)
T_MULE_WEIGHT               = 85    # R14
T_CODE_FARMING_7D           = 20    # R15  referral_code_usage_7d > this
T_CODE_FARMING_WEIGHT       = 75    # R15
T_DEVICE_SIM_LOW            = 0.75  # R16  lower bound of mutation range
T_DEVICE_SIM_HIGH           = 0.99  # R16  upper bound of mutation range
T_DEVICE_MUT_WEIGHT         = 70    # R16
T_THIN_FILE_CREDIT_MONTHS   = 60    # R17  credit_age_months > this
T_THIN_FILE_ACCOUNT_DAYS    = 30    # R17  referee_account_age_days < this
T_THIN_FILE_WEIGHT          = 60    # R17
T_PROMO_GAP_MIN             = 10    # R18  promo_start_gap_min < this
T_PROMO_DORMANCY_DAYS       = 90    # R18  referrer_dormancy_days > this
T_PROMO_WEIGHT              = 65    # R18

# Score normalisation divisor (Section 4.3)
# Set to 100 so that triggering one critical rule (weight 95) alone
# already puts the score at 0.95 — above the BLOCK threshold.
SCORE_NORMALISER = 100.0


# ── RuleEngine class ───────────────────────────────────────────────────────────

class RuleEngine:
    """
    Evaluates all 18 fraud detection rules against a single referral record.

    Rules R04, R13, R15 require dataset-wide lookups.  Call load_dataset(df)
    once before calling evaluate() on any row.
    """

    def __init__(self):
        # Ordered list of rule methods — order does not affect the score
        # but keeping them in R01..R18 order aids debugging and auditability.
        self.rules = [
            self.r01_same_device,
            self.r02_high_velocity_7d,
            self.r03_disposable_email,
            self.r04_ip_reuse,
            self.r05_referral_ring,
            self.r06_geo_mismatch,
            self.r07_new_account_high_referral,
            self.r08_synthetic_identity,
            self.r09_burst_velocity,
            self.r10_geo_velocity_impossible,
            self.r11_address_collision,
            self.r12_no_usage_intent,
            self.r13_identity_doc_reuse,
            self.r14_mule_account_cluster,
            self.r15_referral_code_farming,
            self.r16_device_fingerprint_mutation,
            self.r17_thin_file_credit_mismatch,
            self.r18_promo_timing_exploitation,
        ]

        # Dataset-wide lookup tables — populated by load_dataset()
        # Key: field value  →  Value: count of rows with that value
        self._pan_count:  dict[str, int] = {}   # R13
        self._ip_count:   dict[str, int] = {}   # R04 (backup if column missing)
        self._code_count: dict[str, int] = {}   # R15 (backup if column missing)
        self._dataset_loaded = False

    # ── Dataset-wide lookup builder ────────────────────────────────────────────

    def load_dataset(self, df: pd.DataFrame) -> None:
        """
        Pre-compute lookup tables from the full dataset.
        Must be called once before batch evaluation.

        Rules that need this:
          R13 — pan_hash must appear in more than 1 row → build pan_count map
          R04 — ip_account_count column already in dataset, but we also build
                a backup map from referee_ip in case the column is absent
          R15 — referral_code_usage_7d column already in dataset, but we build
                a backup map from referral_code as well
        """
        # R13: count how many times each pan_hash appears
        self._pan_count = df['pan_hash'].value_counts().to_dict()

        # R04 backup: count accounts per referee IP
        self._ip_count  = df['referee_ip'].value_counts().to_dict()

        # R15 backup: count referral code usage per code
        self._code_count = df['referral_code'].value_counts().to_dict()

        self._dataset_loaded = True
        print(f"[RuleEngine] Lookup tables built from {len(df):,} records.")
        print(f"  Unique PAN hashes : {len(self._pan_count):,}")
        print(f"  Unique referee IPs: {len(self._ip_count):,}")
        print(f"  Unique referral codes: {len(self._code_count):,}")

    # ── Core evaluation method ─────────────────────────────────────────────────

    def evaluate(self, record: dict) -> dict:
        """
        Evaluate all 18 rules against a single referral record.

        Parameters
        ----------
        record : dict
            A single row from the dataset as a Python dict.
            (Use row.to_dict() when iterating a DataFrame.)

        Returns
        -------
        dict with keys:
          rules_triggered  : list[str]  — IDs of rules that fired
          rule_score       : float      — normalised 0–1 risk score
          is_hard_block    : bool       — True if any hard-block rule fired
          explanation      : str        — human-readable explanation string
        """
        triggered    = []
        total_weight = 0

        for rule_fn in self.rules:
            result = rule_fn(record)
            if result['triggered']:
                triggered.append(result['rule_id'])
                total_weight += result['weight']

        # Section 4.3: normalise, cap at 1.0
        rule_score    = min(total_weight / SCORE_NORMALISER, 1.0)
        is_hard_block = bool(HARD_BLOCK_RULES & set(triggered))

        return {
            'rules_triggered': triggered,
            'rule_score':      round(rule_score, 4),
            'is_hard_block':   is_hard_block,
            'explanation':     self._build_explanation(triggered, rule_score,
                                                       is_hard_block),
        }

    # ── Explanation builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_explanation(triggered: list, score: float,
                           hard_block: bool) -> str:
        if not triggered:
            return "No rules triggered. Low risk."
        hard = [r for r in triggered if r in HARD_BLOCK_RULES]
        soft = [r for r in triggered if r not in HARD_BLOCK_RULES]
        parts = []
        if hard:
            parts.append(f"Hard-block rules triggered: {hard}.")
        if soft:
            parts.append(f"Review rules triggered: {soft}.")
        parts.append(f"Rule score: {score:.4f}.")
        if hard_block:
            parts.append("Decision: BLOCK (hard rule override — ML bypassed).")
        return " ".join(parts)

    # ── Helper: safe field access ──────────────────────────────────────────────

    @staticmethod
    def _get(record: dict, key: str, default=None):
        """Safely retrieve a field — handles both dict and pd.Series."""
        try:
            val = record[key]
            # pandas NaN → default
            if val != val:      # NaN != NaN is True
                return default
            return val
        except (KeyError, TypeError):
            return default

    # ══════════════════════════════════════════════════════════════════════════
    # RULE IMPLEMENTATIONS — Section 4.2 Rule Catalogue
    # Each method returns:
    #   {'rule_id': str, 'triggered': bool, 'weight': int}
    # ══════════════════════════════════════════════════════════════════════════

    # ── R01: Same Device Referral  (HARD BLOCK, weight 95) ────────────────────
    def r01_same_device(self, r: dict) -> dict:
        """
        The same physical device is used by both referrer and referee.
        One person referring themselves under a second identity.
        Fraud pattern: FP-01 / SELF_REFERRAL
        Trigger: referrer_device_id == referee_device_id
        """
        referrer_dev = self._get(r, 'referrer_device_id', '')
        referee_dev  = self._get(r, 'referee_device_id',  '')
        triggered    = (referrer_dev == referee_dev
                        and referrer_dev != ''
                        and referrer_dev is not None)
        return {'rule_id': 'R01', 'triggered': triggered,
                'weight': T_SAME_DEVICE_WEIGHT}

    # ── R02: High Referral Velocity (7-day)  (weight 75) ──────────────────────
    def r02_high_velocity_7d(self, r: dict) -> dict:
        """
        Referrer submitted more than 5 referrals in the past 7 days.
        Indicates scripted or batch submission of fake identities.
        Fraud pattern: FP-03 / HIGH_VELOCITY
        Trigger: referrals_in_7d > 5
        """
        count     = self._get(r, 'referrals_in_7d', 0)
        triggered = int(count) > T_HIGH_VELOCITY_7D
        return {'rule_id': 'R02', 'triggered': triggered,
                'weight': T_HIGH_VELOCITY_WEIGHT}

    # ── R03: Disposable Email Domain  (weight 70) ─────────────────────────────
    def r03_disposable_email(self, r: dict) -> dict:
        """
        Referee registered using a temporary / throwaway email domain.
        Strong signal of a non-genuine identity.
        Fraud pattern: FP-04 / DISPOSABLE_EMAIL
        Trigger: email_domain IN DISPOSABLE_DOMAINS
        """
        domain    = str(self._get(r, 'email_domain', '')).lower().strip()
        triggered = domain in DISPOSABLE_DOMAINS
        return {'rule_id': 'R03', 'triggered': triggered,
                'weight': T_DISPOSABLE_EMAIL_WEIGHT}

    # ── R04: IP Address Reuse  (weight 80) ────────────────────────────────────
    def r04_ip_reuse(self, r: dict) -> dict:
        """
        Multiple distinct accounts originate from the same IP address.
        Primary signal: ip_account_count column (pre-computed in dataset).
        Fallback: dataset-wide lookup on referee_ip if column is absent.
        Fraud pattern: FP-05 / IP_REUSE
        Trigger: ip_account_count > 3
        """
        # Prefer the pre-computed column in the dataset
        count = self._get(r, 'ip_account_count', None)
        if count is None and self._dataset_loaded:
            # Fallback: derive from our lookup table
            ip    = self._get(r, 'referee_ip', '')
            count = self._ip_count.get(ip, 1)
        count     = int(count) if count is not None else 1
        triggered = count > T_IP_REUSE_COUNT
        return {'rule_id': 'R04', 'triggered': triggered,
                'weight': T_IP_REUSE_WEIGHT}

    # ── R05: Referral Ring Detection  (HARD BLOCK, weight 90) ─────────────────
    def r05_referral_ring(self, r: dict) -> dict:
        """
        Record belongs to a cluster where accounts refer each other
        in a circular pattern to collect rewards without genuine acquisition.
        A cluster_size > 3 means the record is part of a multi-node ring.
        Fraud pattern: FP-02 / REFERRAL_RING
        Trigger: referral_cluster_size > 3
        """
        size      = self._get(r, 'referral_cluster_size', 1)
        triggered = int(size) > T_RING_CLUSTER_SIZE
        return {'rule_id': 'R05', 'triggered': triggered,
                'weight': T_RING_WEIGHT}

    # ── R06: Geographic Risk Pattern  (weight 60) ─────────────────────────────
    def r06_geo_mismatch(self, r: dict) -> dict:
        """
        Referrer's last known city does not match the referee's application city.
        Alone this is a soft signal; it gains strength combined with R10.
        Fraud pattern: FP-08 (partial) / GEO_VELOCITY
        Trigger: location_mismatch_flag == True
        """
        flag      = self._get(r, 'location_mismatch_flag', False)
        triggered = bool(flag)
        return {'rule_id': 'R06', 'triggered': triggered,
                'weight': T_GEO_MISMATCH_WEIGHT}

    # ── R07: New Account + High Referral Activity  (weight 65) ────────────────
    def r07_new_account_high_referral(self, r: dict) -> dict:
        """
        A very new referrer account (< 3 days old) is already submitting
        multiple referrals — a strong signal of a purpose-built mule account.
        Fraud pattern: FP-03 / HIGH_VELOCITY (early-stage variant)
        Trigger: referrer_account_age_days < 3 AND referrals_in_7d > 2
        """
        age       = self._get(r, 'referrer_account_age_days', 9999)
        refs      = self._get(r, 'referrals_in_7d', 0)
        triggered = (int(age) < T_NEW_ACCOUNT_AGE_DAYS
                     and int(refs) > T_NEW_ACCOUNT_REFERRALS)
        return {'rule_id': 'R07', 'triggered': triggered,
                'weight': T_NEW_ACCOUNT_WEIGHT}

    # ── R08: Synthetic Identity Signal  (weight 70) ───────────────────────────
    def r08_synthetic_identity(self, r: dict) -> dict:
        """
        High identity risk score indicating a fabricated or partially
        fabricated identity — combination of real and false personal details.
        Fraud pattern: FP-06 / SYNTHETIC_IDENTITY
        Trigger: identity_risk_score > 0.7
        """
        score     = self._get(r, 'identity_risk_score', 0.0)
        triggered = float(score) > T_IDENTITY_RISK_SCORE
        return {'rule_id': 'R08', 'triggered': triggered,
                'weight': T_IDENTITY_RISK_WEIGHT}

    # ── R09: Burst Referral Velocity  (weight 80) ─────────────────────────────
    def r09_burst_velocity(self, r: dict) -> dict:
        """
        Multiple referrals submitted within a very short window — indicates
        automated / scripted submission rather than genuine individual referrals.
        Fraud pattern: FP-07 / BURST_VELOCITY
        Trigger: referrals_in_1hr > 3  OR  min_gap_sec < 120
        (Either condition alone is sufficient.)
        """
        count_1hr = self._get(r, 'referrals_in_1hr', 0)
        gap_sec   = self._get(r, 'min_gap_sec', 99999)
        triggered = (int(count_1hr) > T_BURST_1HR
                     or int(gap_sec)   < T_BURST_GAP_SEC)
        return {'rule_id': 'R09', 'triggered': triggered,
                'weight': T_BURST_WEIGHT}

    # ── R10: Geo-Velocity Impossibility  (HARD BLOCK, weight 85) ──────────────
    def r10_geo_velocity_impossible(self, r: dict) -> dict:
        """
        Account appears in two geographically distant locations within a time
        window that makes travel physically impossible (> 900 km/h implies
        supersonic flight or account sharing / takeover).
        Fraud pattern: FP-08 / GEO_VELOCITY
        Trigger: geo_velocity_kmph > 900
        """
        velocity  = self._get(r, 'geo_velocity_kmph', 0.0)
        triggered = float(velocity) > T_GEO_VELOCITY_KMPH
        return {'rule_id': 'R10', 'triggered': triggered,
                'weight': T_GEO_VELOCITY_WEIGHT}

    # ── R11: Address Collision  (weight 55) ───────────────────────────────────
    def r11_address_collision(self, r: dict) -> dict:
        """
        More than 2 accounts registered at the same physical address within
        30 days — indicates one person creating multiple identities.
        Note: intentionally low weight (55) to minimise false positives for
        legitimate family members sharing an address.
        Fraud pattern: FP-09 / ADDRESS_COLLISION
        Trigger: same_address_count > 2
        """
        count     = self._get(r, 'same_address_count', 1)
        triggered = int(count) > T_ADDRESS_COUNT
        return {'rule_id': 'R11', 'triggered': triggered,
                'weight': T_ADDRESS_WEIGHT}

    # ── R12: No Post-Approval Usage Intent  (weight 65) ───────────────────────
    def r12_no_usage_intent(self, r: dict) -> dict:
        """
        Card issued but not genuinely used — referral was made only to claim
        the reward, not to acquire a real customer.
        Post-issuance data is pre-populated in the synthetic dataset.
        Fraud pattern: FP-10 / NO_USAGE_INTENT
        Trigger: card_activated == False  OR  cash_advance_ratio > 0.8
        """
        activated = self._get(r, 'card_activated', True)
        ca_ratio  = self._get(r, 'cash_advance_ratio', 0.0)
        triggered = (not bool(activated)
                     or float(ca_ratio) > T_CASH_ADVANCE_RATIO)
        return {'rule_id': 'R12', 'triggered': triggered,
                'weight': T_NO_USAGE_WEIGHT}

    # ── R13: Identity Document Reuse  (HARD BLOCK, weight 95) ─────────────────
    def r13_identity_doc_reuse(self, r: dict) -> dict:
        """
        The same government identity document (PAN / SSN) appears in more
        than one application — near-certain fraud signal.

        IMPORTANT: This is a dataset-wide rule.  It uses the _pan_count
        lookup table built by load_dataset().  Without that call, it always
        returns triggered=False and logs a warning.

        Fraud pattern: FP-11 / IDENTITY_DOC_REUSE
        Trigger: pan_hash appears in > 1 application across the dataset
        """
        if not self._dataset_loaded:
            # Cannot evaluate without dataset context — fail safe (no trigger)
            return {'rule_id': 'R13', 'triggered': False,
                    'weight': T_PAN_REUSE_WEIGHT}

        pan       = self._get(r, 'pan_hash', '')
        count     = self._pan_count.get(str(pan), 0)
        triggered = count > T_PAN_REUSE_COUNT      # seen in > 1 row
        return {'rule_id': 'R13', 'triggered': triggered,
                'weight': T_PAN_REUSE_WEIGHT}

    # ── R14: Mule Account Cluster  (HARD BLOCK, weight 85) ───────────────────
    def r14_mule_account_cluster(self, r: dict) -> dict:
        """
        Very new accounts (< 7 days old) that are already part of a referral
        ring cluster (size > 3).  These 'mule' accounts are purpose-built to
        participate in a ring and then be abandoned.
        Fraud pattern: FP-02 + FP-03 / REFERRAL_RING + new account signal
        Trigger: referee_account_age_days < 7  AND  referral_cluster_size > 3
        """
        age       = self._get(r, 'referee_account_age_days', 9999)
        size      = self._get(r, 'referral_cluster_size', 1)
        triggered = (int(age)  < T_MULE_ACCOUNT_AGE
                     and int(size) > T_MULE_CLUSTER_SIZE)
        return {'rule_id': 'R14', 'triggered': triggered,
                'weight': T_MULE_WEIGHT}

    # ── R15: Referral Code Farming  (weight 75) ───────────────────────────────
    def r15_referral_code_farming(self, r: dict) -> dict:
        """
        A referral code is shared publicly (forums, Telegram groups) and used
        by many unrelated people with no genuine relationship to the referrer.
        Primary signal: referral_code_usage_7d column (pre-computed).
        Fallback: dataset-wide lookup on referral_code.
        Fraud pattern: FP-12 / REFERRAL_CODE_FARMING
        Trigger: referral_code_usage_7d > 20
        """
        # Prefer pre-computed column
        usage = self._get(r, 'referral_code_usage_7d', None)
        if usage is None and self._dataset_loaded:
            code  = self._get(r, 'referral_code', '')
            usage = self._code_count.get(str(code), 1)
        usage     = int(usage) if usage is not None else 1
        triggered = usage > T_CODE_FARMING_7D
        return {'rule_id': 'R15', 'triggered': triggered,
                'weight': T_CODE_FARMING_WEIGHT}

    # ── R16: Device Fingerprint Mutation  (weight 70) ─────────────────────────
    def r16_device_fingerprint_mutation(self, r: dict) -> dict:
        """
        Device fingerprint similarity between 0.75 and 0.99 — the device is
        'almost' but not exactly matching a known device on the same IP.
        This window catches fraudsters who slightly alter browser attributes
        (user-agent, fonts, screen res) to evade exact-match detection.
        A score of exactly 1.0 is caught by R01 (same device).
        A score below 0.75 is genuinely different — not suspicious.
        Fraud pattern: FP (advanced) / device evasion
        Trigger: 0.75 < device_similarity_score < 0.99
        """
        score     = self._get(r, 'device_similarity_score', 0.0)
        score     = float(score)
        triggered = T_DEVICE_SIM_LOW < score < T_DEVICE_SIM_HIGH
        return {'rule_id': 'R16', 'triggered': triggered,
                'weight': T_DEVICE_MUT_WEIGHT}

    # ── R17: Thin-File Credit Mismatch  (weight 60) ───────────────────────────
    def r17_thin_file_credit_mismatch(self, r: dict) -> dict:
        """
        A referee whose account is very new (< 30 days) but who claims a
        credit history older than 60 months.  Real people cannot build a
        60-month credit history in under 30 days — this is a synthetic
        identity signal.
        Fraud pattern: FP-06 / SYNTHETIC_IDENTITY (credit variant)
        Trigger: credit_age_months > 60  AND  referee_account_age_days < 30
        """
        credit_age  = self._get(r, 'credit_age_months',         0)
        account_age = self._get(r, 'referee_account_age_days', 9999)
        triggered   = (int(credit_age)  > T_THIN_FILE_CREDIT_MONTHS
                       and int(account_age) < T_THIN_FILE_ACCOUNT_DAYS)
        return {'rule_id': 'R17', 'triggered': triggered,
                'weight': T_THIN_FILE_WEIGHT}

    # ── R18: Promo Timing Exploitation  (weight 65) ───────────────────────────
    def r18_promo_timing_exploitation(self, r: dict) -> dict:
        """
        A dormant account (no activity for > 90 days) springs to life
        within 10 minutes of a promotional offer starting.
        Fraudsters monitor referral programs and automate submission at
        the exact moment a high-reward promotion activates.
        Fraud pattern: FP-12 (timing variant) / REFERRAL_CODE_FARMING
        Trigger: promo_active == True
                 AND promo_start_gap_min in [0, 10)
                 AND referrer_dormancy_days > 90
        """
        promo_active  = self._get(r, 'promo_active',            False)
        gap_min       = self._get(r, 'promo_start_gap_min',     9999)
        dormancy_days = self._get(r, 'referrer_dormancy_days',  0)
        triggered     = (bool(promo_active)
                         and 0 <= int(gap_min)       < T_PROMO_GAP_MIN
                         and int(dormancy_days)       > T_PROMO_DORMANCY_DAYS)
        return {'rule_id': 'R18', 'triggered': triggered,
                'weight': T_PROMO_WEIGHT}


# ══════════════════════════════════════════════════════════════════════════════
# Batch evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(df: pd.DataFrame,
                     engine: RuleEngine | None = None) -> pd.DataFrame:
    """
    Convenience function: apply RuleEngine to every row of df and add
    result columns in-place.

    Returns a copy of df with three new columns:
      rule_score       : float  — normalised rule-based risk score (0–1)
      rules_triggered  : str    — comma-separated list of triggered rule IDs
      is_hard_block    : bool   — True if any hard-block rule fired
      rule_explanation : str    — human-readable explanation

    Parameters
    ----------
    df     : pd.DataFrame — the full referral dataset
    engine : RuleEngine   — optional pre-built engine; created if None
    """
    if engine is None:
        engine = RuleEngine()

    engine.load_dataset(df)

    results = df.apply(lambda row: engine.evaluate(row.to_dict()), axis=1)

    out = df.copy()
    out['rule_score']       = results.apply(lambda x: x['rule_score'])
    out['rules_triggered']  = results.apply(
        lambda x: ','.join(x['rules_triggered']) if x['rules_triggered'] else 'NONE'
    )
    out['is_hard_block']    = results.apply(lambda x: x['is_hard_block'])
    out['rule_explanation'] = results.apply(lambda x: x['explanation'])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Standalone evaluation — run this file directly to see results on the dataset
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, precision_score,
                                 recall_score, f1_score)

    DATA_PATH = os.path.join('data', 'referral_dataset.csv')

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run  python src/generate_dataset.py  first.")
        raise SystemExit(1)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} records loaded.\n")

    # ── Run rule engine on full dataset ───────────────────────────────────────
    print("Running rule engine on full dataset...")
    df_scored = evaluate_dataset(df)

    # ── Threshold: rule_score > 0.5 → predicted fraud ─────────────────────────
    threshold  = 0.5
    y_true     = df_scored['is_fraud']
    y_pred     = (df_scored['rule_score'] > threshold).astype(int)
    y_score    = df_scored['rule_score']

    # ── Section 4.5: Preliminary Rule Engine Evaluation ──────────────────────
    print("\n" + "="*60)
    print("RULE ENGINE — PRELIMINARY EVALUATION (Section 4.5)")
    print("="*60)
    print(f"\nClassification threshold : rule_score > {threshold}")
    print()
    print(classification_report(y_true, y_pred,
                                 target_names=['Legitimate', 'Fraud']))

    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_score)

    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"AUC-ROC          : {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negative  (legit correctly approved) : {cm[0][0]:>6,}")
    print(f"  False Positive (legit wrongly blocked)    : {cm[0][1]:>6,}")
    print(f"  False Negative (fraud missed)             : {cm[1][0]:>6,}")
    print(f"  True Positive  (fraud correctly blocked)  : {cm[1][1]:>6,}")

    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"\nFalse Positive Rate : {fpr:.4f}")

    # ── Rule trigger frequency ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RULE TRIGGER FREQUENCY ANALYSIS")
    print("="*60)
    fraud_rows = df_scored[df_scored['is_fraud'] == 1]

    rule_counts: dict[str, int] = {}
    for rules_str in fraud_rows['rules_triggered']:
        if rules_str != 'NONE':
            for r in rules_str.split(','):
                rule_counts[r.strip()] = rule_counts.get(r.strip(), 0) + 1

    total_fraud = len(fraud_rows)
    print(f"\nTotal fraud records: {total_fraud}")
    print(f"{'Rule':<6}  {'Count':>6}  {'% of Fraud':>10}  Description")
    print("-"*60)
    desc = {
        'R01': 'Same Device Referral',
        'R02': 'High Velocity (7d)',
        'R03': 'Disposable Email',
        'R04': 'IP Address Reuse',
        'R05': 'Referral Ring',
        'R06': 'Geo Mismatch',
        'R07': 'New Acct + High Referral',
        'R08': 'Synthetic Identity',
        'R09': 'Burst Velocity',
        'R10': 'Geo-Velocity Impossibility',
        'R11': 'Address Collision',
        'R12': 'No Usage Intent',
        'R13': 'Identity Document Reuse',
        'R14': 'Mule Account Cluster',
        'R15': 'Code Farming',
        'R16': 'Device Fingerprint Mutation',
        'R17': 'Thin-File Credit Mismatch',
        'R18': 'Promo Timing Exploitation',
    }
    for rule_id in [f'R{i:02d}' for i in range(1, 19)]:
        count = rule_counts.get(rule_id, 0)
        pct   = count / total_fraud * 100 if total_fraud > 0 else 0
        print(f"  {rule_id:<6}  {count:>6}  {pct:>9.1f}%  {desc.get(rule_id,'')}")

    # ── Hard block analysis ────────────────────────────────────────────────────
    hard_blocks = df_scored['is_hard_block'].sum()
    hb_fraud    = df_scored[df_scored['is_fraud'] == 1]['is_hard_block'].sum()
    hb_legit    = df_scored[df_scored['is_fraud'] == 0]['is_hard_block'].sum()
    print(f"\nHard-block breakdown:")
    print(f"  Total hard blocks         : {hard_blocks:,}")
    print(f"  Hard blocks on fraud rows : {hb_fraud:,}  "
          f"({hb_fraud/total_fraud*100:.1f}% of fraud caught by hard block)")
    print(f"  Hard blocks on legit rows : {hb_legit:,}  "
          f"(false positives from hard-block rules)")

    # ── Save scored dataset ───────────────────────────────────────────────────
    scored_path = os.path.join('data', 'referral_dataset_scored.csv')
    df_scored.to_csv(scored_path, index=False)
    print(f"\nScored dataset saved to: {scored_path}")
    print("Columns added: rule_score, rules_triggered, is_hard_block, rule_explanation")
