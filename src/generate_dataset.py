"""
generate_dataset.py
====================
Synthetic Referral Fraud Dataset Generator
Dissertation: Fraud Detection in Credit Card Referral Systems
            Using Hybrid Rule-Based and Machine Learning Approach
BITS Pilani WILP — M.Tech Software Systems

Generates 10,026 records across 12 fraud types + legitimate records.
All 18 rules in rule_engine.py are detectable from the columns this
script produces.

Columns added beyond the original roadmap snippet (required for rules
R04, R06, R12, R15):
  - ip_account_count        : int   — how many accounts share this IP
  - location_mismatch_flag  : bool  — referrer city != referee city
  - card_activated          : bool  — whether card was activated post-issuance
  - cash_advance_ratio      : float — ratio of cash advances to total spend
  - referral_code_usage_7d  : int   — how many times this referral code was
                                      used in the past 7 days

Usage:
  python generate_dataset.py
  → writes data/referral_dataset.csv
"""

import os
import random
import hashlib
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)
fake = Faker('en_IN')
Faker.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
DISPOSABLE_DOMAINS = [
    'mailinator.com', 'guerrillamail.com', 'throwaway.email',
    'temp-mail.org', 'fakeinbox.com', 'yopmail.com', 'maildrop.cc',
    'trashmail.com', 'sharklasers.com', 'dispostable.com'
]

LEGIT_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
    'rediffmail.com', 'icloud.com', 'protonmail.com'
]

INDIAN_CITIES = [
    'Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Hyderabad',
    'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat'
]

# 3 promotional windows in 2024
PROMO_PERIODS = [
    (datetime(2024, 1, 1),  datetime(2024, 1, 15)),
    (datetime(2024, 3, 1),  datetime(2024, 3, 10)),
    (datetime(2024, 6, 1),  datetime(2024, 6, 7)),
]

# Fraud type counts — target ~18% fraud (1800 / 10026)
# Each number is how many records to generate for that fraud type
FRAUD_COUNTS = {
    'SELF_REFERRAL':          360,   # FP-01 / R01
    'REFERRAL_RING':          270,   # FP-02 / R05
    'HIGH_VELOCITY':          216,   # FP-03 / R02, R09
    'DISPOSABLE_EMAIL':       180,   # FP-04 / R03
    'IP_REUSE':               162,   # FP-05 / R04
    'SYNTHETIC_IDENTITY':     144,   # FP-06 / R08, R17
    'BURST_VELOCITY':         126,   # FP-07 / R09
    'GEO_VELOCITY':           108,   # FP-08 / R10
    'ADDRESS_COLLISION':       90,   # FP-09 / R11
    'IDENTITY_DOC_REUSE':      72,   # FP-11 / R13
    'REFERRAL_CODE_FARMING':   54,   # FP-12 / R15
    'NO_USAGE_INTENT':         18,   # FP-10 / R12
}
# Legitimate records fill the rest up to 10026
TOTAL_RECORDS  = 10026
TOTAL_FRAUD    = sum(FRAUD_COUNTS.values())          # 1800
TOTAL_LEGIT    = TOTAL_RECORDS - TOTAL_FRAUD         # 8226


# ── Helper utilities ───────────────────────────────────────────────────────────

def is_promo_time(ts: datetime):
    """Return (promo_active, minutes_since_promo_start)."""
    for start, end in PROMO_PERIODS:
        if start <= ts <= end:
            gap = int((ts - start).total_seconds() // 60)
            return True, gap
    return False, -1


def make_device_id():
    return f'DEV_{fake.md5()[:8].upper()}'


def make_referral_code(referrer_id: str):
    return f'CODE_{referrer_id[-4:]}_{random.randint(100,999)}'


def make_pan_hash(seed: str = None):
    raw = seed if seed else fake.bothify('????######')
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_cluster_id(n: int):
    return f'CLUSTER_{n:04d}'


def pick_city():
    return random.choice(INDIAN_CITIES)


def make_referrer_pool(size: int = 500) -> list:
    """
    Pre-build a pool of referrer accounts so we can realistically
    reuse referrer_id, device, IP, code across multiple referral records.
    """
    pool = []
    for _ in range(size):
        rid = f'CUST_{random.randint(10000, 59999)}'
        pool.append({
            'id':     rid,
            'device': make_device_id(),
            'ip':     fake.ipv4_private(),
            'code':   make_referral_code(rid),
            'city':   pick_city(),
        })
    return pool


# ── Base record skeleton ───────────────────────────────────────────────────────

def _base(i: int, referrer: dict, ts: datetime, fraud_type: str,
          is_fraud: int) -> dict:
    """
    Shared fields present in every record.
    Fraud-specific generators override individual fields after calling this.
    """
    promo, gap = is_promo_time(ts)
    return {
        # ── Identity ──────────────────────────────────────────────────
        'referral_id':               f'REF_{i:06d}',
        'referrer_id':               referrer['id'],
        'referee_id':                f'CUST_{random.randint(60000, 99999)}',
        'referral_code':             referrer['code'],
        'timestamp':                 ts,

        # ── Device & network ──────────────────────────────────────────
        'referrer_device_id':        referrer['device'],
        'referee_device_id':         make_device_id(),          # unique by default
        'referrer_ip':               referrer['ip'],
        'referee_ip':                fake.ipv4_private(),        # unique by default
        'ip_account_count':          random.randint(1, 2),      # R04: legit default

        # ── Email ─────────────────────────────────────────────────────
        'email_domain':              random.choice(LEGIT_DOMAINS),

        # ── Account age ───────────────────────────────────────────────
        'referrer_account_age_days': random.randint(180, 2000),
        'referee_account_age_days':  random.randint(30, 365),

        # ── Velocity / timing ─────────────────────────────────────────
        'referrals_in_7d':           random.randint(0, 3),
        'referrals_in_1hr':          random.randint(0, 1),
        'min_gap_sec':               random.randint(3600, 86400),
        'referral_code_usage_7d':    random.randint(1, 5),      # R15: legit default

        # ── Geography ─────────────────────────────────────────────────
        'referrer_city':             referrer['city'],
        'referee_city':              pick_city(),
        'geo_velocity_kmph':         round(random.uniform(0, 50), 2),
        'location_mismatch_flag':    False,                      # R06: default

        # ── Address ───────────────────────────────────────────────────
        'same_address_count':        random.randint(1, 2),

        # ── Identity / credit ─────────────────────────────────────────
        'pan_hash':                  make_pan_hash(),
        'identity_risk_score':       round(random.uniform(0.0, 0.3), 3),
        'credit_age_months':         random.randint(12, 120),
        'device_similarity_score':   round(random.uniform(0.0, 0.5), 3),

        # ── Post-issuance behaviour ───────────────────────────────────
        'card_activated':            True,                       # R12: legit default
        'cash_advance_ratio':        round(random.uniform(0.0, 0.2), 3),

        # ── Promotional context ───────────────────────────────────────
        'promo_active':              promo,
        'promo_start_gap_min':       gap,
        'referrer_dormancy_days':    random.randint(0, 30),

        # ── Ring / cluster ────────────────────────────────────────────
        'referral_cluster_id':       'CLUSTER_0000',
        'referral_cluster_size':     1,

        # ── Labels ────────────────────────────────────────────────────
        'fraud_type':                fraud_type,
        'is_fraud':                  is_fraud,
    }


# ── 1. LEGITIMATE records (8226) ───────────────────────────────────────────────

def generate_legitimate(i: int, referrer_pool: list) -> dict:
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'LEGITIMATE', 0)

    # Ensure legitimate records DON'T trigger any fraud rules
    rec['referrer_account_age_days'] = random.randint(180, 2000)
    rec['referee_account_age_days']  = random.randint(30, 365)
    rec['referrals_in_7d']           = random.randint(0, 3)
    rec['referrals_in_1hr']          = 0
    rec['min_gap_sec']               = random.randint(7200, 86400)
    rec['geo_velocity_kmph']         = round(random.uniform(0, 60), 2)
    rec['same_address_count']        = random.randint(1, 2)
    rec['identity_risk_score']       = round(random.uniform(0.0, 0.3), 3)
    rec['credit_age_months']         = random.randint(12, 96)
    rec['device_similarity_score']   = round(random.uniform(0.0, 0.5), 3)
    rec['ip_account_count']          = random.randint(1, 2)
    rec['referral_code_usage_7d']    = random.randint(1, 5)
    rec['card_activated']            = True
    rec['cash_advance_ratio']        = round(random.uniform(0.0, 0.15), 3)
    rec['location_mismatch_flag']    = False
    return rec


# ── 2. FRAUD TYPE GENERATORS ───────────────────────────────────────────────────

# ── FP-01: SELF_REFERRAL — same device (R01) ──────────────────────────────────
def generate_self_referral(i: int, referrer_pool: list) -> dict:
    """
    Same device_id for both referrer and referee.
    The person refers themselves using a second identity.
    Triggers: R01 (hard block)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'SELF_REFERRAL', 1)

    shared_device = referrer['device']          # same device both sides
    rec['referee_device_id']        = shared_device
    rec['referrer_device_id']       = shared_device
    # Make other fields look mostly normal to test R01 in isolation
    rec['referrals_in_7d']          = random.randint(1, 4)
    rec['identity_risk_score']      = round(random.uniform(0.1, 0.4), 3)
    return rec


# ── FP-02: REFERRAL_RING (R05) ────────────────────────────────────────────────
def generate_referral_ring(i: int, referrer_pool: list,
                           cluster_id: str, cluster_size: int) -> dict:
    """
    Group of accounts referring each other in a circle.
    All records in the same ring share a cluster_id and cluster_size > 3.
    Triggers: R05, R14 (if accounts are new)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-6m', end_date='now')
    rec = _base(i, referrer, ts, 'REFERRAL_RING', 1)

    rec['referral_cluster_id']      = cluster_id
    rec['referral_cluster_size']    = cluster_size           # > 3
    rec['referee_account_age_days'] = random.randint(5, 45)  # relatively new
    rec['referrer_account_age_days']= random.randint(5, 60)
    rec['referrals_in_7d']          = random.randint(2, 6)
    rec['ip_account_count']         = random.randint(1, 3)
    return rec


# ── FP-03: HIGH_VELOCITY — many referrals over 7 days (R02) ──────────────────
def generate_high_velocity(i: int, referrer_pool: list) -> dict:
    """
    One referrer submits many referrals in a 7-day window.
    Triggers: R02. May co-trigger R09 if gap is also small.
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'HIGH_VELOCITY', 1)

    rec['referrals_in_7d']          = random.randint(6, 20)
    rec['referrals_in_1hr']         = random.randint(1, 3)
    rec['min_gap_sec']              = random.randint(300, 3600)
    rec['referrer_account_age_days']= random.randint(30, 400)
    rec['identity_risk_score']      = round(random.uniform(0.1, 0.5), 3)
    return rec


# ── FP-04: DISPOSABLE_EMAIL (R03) ─────────────────────────────────────────────
def generate_disposable_email(i: int, referrer_pool: list) -> dict:
    """
    Referee uses a throwaway email domain.
    Triggers: R03
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'DISPOSABLE_EMAIL', 1)

    rec['email_domain']             = random.choice(DISPOSABLE_DOMAINS)
    rec['referee_account_age_days'] = random.randint(1, 10)   # brand new account
    rec['identity_risk_score']      = round(random.uniform(0.3, 0.6), 3)
    return rec


# ── FP-05: IP_REUSE (R04) ─────────────────────────────────────────────────────
def generate_ip_reuse(i: int, referrer_pool: list,
                      shared_ip: str) -> dict:
    """
    Multiple accounts submit applications from the same IP.
    ip_account_count > 3 triggers R04.
    Triggers: R04
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'IP_REUSE', 1)

    rec['referee_ip']               = shared_ip
    rec['referrer_ip']              = shared_ip
    rec['ip_account_count']         = random.randint(4, 12)
    rec['referee_account_age_days'] = random.randint(1, 20)
    return rec


# ── FP-06: SYNTHETIC_IDENTITY (R08, R17) ─────────────────────────────────────
def generate_synthetic_identity(i: int, referrer_pool: list) -> dict:
    """
    Fabricated identity. High identity_risk_score + credit history that
    is too old for the account age (thin-file mismatch).
    Triggers: R08 (identity_risk_score > 0.7), R17 (credit_age > 60 AND
              account_age < 30)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'SYNTHETIC_IDENTITY', 1)

    rec['identity_risk_score']      = round(random.uniform(0.71, 0.99), 3)
    rec['credit_age_months']        = random.randint(61, 180)  # R17: suspiciously old
    rec['referee_account_age_days'] = random.randint(1, 25)    # R17: very new account
    rec['email_domain']             = random.choice(
        DISPOSABLE_DOMAINS + LEGIT_DOMAINS)                    # mixed signals
    rec['device_similarity_score']  = round(random.uniform(0.3, 0.6), 3)
    return rec


# ── FP-07: BURST_VELOCITY — referrals within minutes (R09) ───────────────────
def generate_burst_velocity(i: int, referrer_pool: list) -> dict:
    """
    Multiple referrals submitted within a very short window (< 2 min apart).
    Indicates scripted / automated submission.
    Triggers: R09 (min_gap_sec < 120 AND/OR referrals_in_1hr > 3)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'BURST_VELOCITY', 1)

    rec['referrals_in_1hr']         = random.randint(4, 15)
    rec['min_gap_sec']              = random.randint(10, 110)  # < 120 sec
    rec['referrals_in_7d']          = random.randint(5, 15)
    rec['referrer_dormancy_days']   = random.randint(0, 10)
    return rec


# ── FP-08: GEO_VELOCITY — impossible travel (R10) ────────────────────────────
def generate_geo_velocity(i: int, referrer_pool: list) -> dict:
    """
    Account appears in two geographically distant locations within minutes.
    geo_velocity_kmph > 900 is physically impossible.
    Triggers: R10 (hard block)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'GEO_VELOCITY', 1)

    rec['geo_velocity_kmph']        = round(random.uniform(901, 2500), 2)
    rec['location_mismatch_flag']   = True                     # R06 co-trigger
    # Two distant cities — Mumbai to New York equivalent scenario
    rec['referrer_city']            = 'Mumbai'
    rec['referee_city']             = 'Delhi'                  # used for display
    rec['referrer_ip']              = fake.ipv4_public()
    rec['referee_ip']               = fake.ipv4_public()
    return rec


# ── FP-09: ADDRESS_COLLISION (R11) ────────────────────────────────────────────
def generate_address_collision(i: int, referrer_pool: list) -> dict:
    """
    Many accounts registered at the same physical address in a short period.
    same_address_count > 2 triggers R11 (review, not hard block).
    Triggers: R11
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'ADDRESS_COLLISION', 1)

    rec['same_address_count']       = random.randint(3, 8)
    rec['referee_account_age_days'] = random.randint(1, 30)
    rec['identity_risk_score']      = round(random.uniform(0.2, 0.5), 3)
    return rec


# ── FP-11: IDENTITY_DOC_REUSE (R13) ──────────────────────────────────────────
def generate_identity_doc_reuse(i: int, referrer_pool: list,
                                shared_pan: str) -> dict:
    """
    The same PAN / SSN hash appears in more than one application.
    Near-certain fraud signal.
    Triggers: R13 (hard block — detected via dataset-wide lookup in rule engine)
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'IDENTITY_DOC_REUSE', 1)

    rec['pan_hash']                 = shared_pan               # deliberately reused
    rec['referee_device_id']        = make_device_id()         # different device
    rec['referee_ip']               = fake.ipv4_private()      # different IP
    rec['referee_account_age_days'] = random.randint(1, 60)
    rec['identity_risk_score']      = round(random.uniform(0.4, 0.8), 3)
    return rec


# ── FP-12: REFERRAL_CODE_FARMING (R15) ───────────────────────────────────────
def generate_referral_code_farming(i: int, referrer_pool: list) -> dict:
    """
    A referral code is shared publicly and used by many unrelated people.
    referral_code_usage_7d > 20 triggers R15.
    Triggers: R15
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'REFERRAL_CODE_FARMING', 1)

    rec['referral_code_usage_7d']   = random.randint(21, 60)
    rec['referrals_in_7d']          = random.randint(6, 20)
    # Referrer was dormant, then suddenly active at promo time
    rec['referrer_dormancy_days']   = random.randint(60, 200)
    rec['promo_active']             = True
    rec['promo_start_gap_min']      = random.randint(0, 120)
    # Geographically diverse referees
    rec['location_mismatch_flag']   = True
    return rec


# ── FP-10: NO_USAGE_INTENT — post-issuance (R12) ─────────────────────────────
def generate_no_usage_intent(i: int, referrer_pool: list) -> dict:
    """
    Card is issued but never genuinely used — referral was only for the reward.
    card_activated = False  OR  cash_advance_ratio > 0.8
    Triggers: R12
    Note: This is a post-issuance pattern. In the dataset it is pre-populated
    to simulate what would be observed after card issuance.
    """
    referrer = random.choice(referrer_pool)
    ts = fake.date_time_between(start_date='-1y', end_date='now')
    rec = _base(i, referrer, ts, 'NO_USAGE_INTENT', 1)

    # Either not activated at all, or heavily cash-advance focused
    if random.random() < 0.6:
        rec['card_activated']       = False
        rec['cash_advance_ratio']   = 0.0
    else:
        rec['card_activated']       = True
        rec['cash_advance_ratio']   = round(random.uniform(0.81, 1.0), 3)

    rec['referee_account_age_days'] = random.randint(30, 120)
    rec['identity_risk_score']      = round(random.uniform(0.1, 0.4), 3)
    return rec


# ── Dataset assembler ──────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    referrer_pool = make_referrer_pool(500)
    records = []
    idx = 1

    # ── Legitimate ────────────────────────────────────────────────────────────
    print(f"Generating {TOTAL_LEGIT} legitimate records...")
    for _ in range(TOTAL_LEGIT):
        records.append(generate_legitimate(idx, referrer_pool))
        idx += 1

    # ── FP-01: Self-Referral ──────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['SELF_REFERRAL']} SELF_REFERRAL records...")
    for _ in range(FRAUD_COUNTS['SELF_REFERRAL']):
        records.append(generate_self_referral(idx, referrer_pool))
        idx += 1

    # ── FP-02: Referral Ring ──────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['REFERRAL_RING']} REFERRAL_RING records...")
    ring_count  = FRAUD_COUNTS['REFERRAL_RING']
    cluster_num = 1
    generated   = 0
    while generated < ring_count:
        size = random.randint(4, 8)                         # ring size 4–8
        size = min(size, ring_count - generated)
        cid  = make_cluster_id(cluster_num)
        for _ in range(size):
            records.append(generate_referral_ring(idx, referrer_pool, cid, size))
            idx       += 1
            generated += 1
        cluster_num += 1

    # ── FP-03: High Velocity ──────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['HIGH_VELOCITY']} HIGH_VELOCITY records...")
    for _ in range(FRAUD_COUNTS['HIGH_VELOCITY']):
        records.append(generate_high_velocity(idx, referrer_pool))
        idx += 1

    # ── FP-04: Disposable Email ───────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['DISPOSABLE_EMAIL']} DISPOSABLE_EMAIL records...")
    for _ in range(FRAUD_COUNTS['DISPOSABLE_EMAIL']):
        records.append(generate_disposable_email(idx, referrer_pool))
        idx += 1

    # ── FP-05: IP Reuse ───────────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['IP_REUSE']} IP_REUSE records...")
    # Each shared_ip is reused across a batch of 4–10 records
    ip_generated = 0
    while ip_generated < FRAUD_COUNTS['IP_REUSE']:
        shared_ip  = fake.ipv4_private()
        batch_size = random.randint(4, 10)
        batch_size = min(batch_size, FRAUD_COUNTS['IP_REUSE'] - ip_generated)
        for _ in range(batch_size):
            records.append(generate_ip_reuse(idx, referrer_pool, shared_ip))
            idx          += 1
            ip_generated += 1

    # ── FP-06: Synthetic Identity ─────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['SYNTHETIC_IDENTITY']} SYNTHETIC_IDENTITY records...")
    for _ in range(FRAUD_COUNTS['SYNTHETIC_IDENTITY']):
        records.append(generate_synthetic_identity(idx, referrer_pool))
        idx += 1

    # ── FP-07: Burst Velocity ─────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['BURST_VELOCITY']} BURST_VELOCITY records...")
    for _ in range(FRAUD_COUNTS['BURST_VELOCITY']):
        records.append(generate_burst_velocity(idx, referrer_pool))
        idx += 1

    # ── FP-08: Geo-Velocity Impossibility ─────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['GEO_VELOCITY']} GEO_VELOCITY records...")
    for _ in range(FRAUD_COUNTS['GEO_VELOCITY']):
        records.append(generate_geo_velocity(idx, referrer_pool))
        idx += 1

    # ── FP-09: Address Collision ──────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['ADDRESS_COLLISION']} ADDRESS_COLLISION records...")
    for _ in range(FRAUD_COUNTS['ADDRESS_COLLISION']):
        records.append(generate_address_collision(idx, referrer_pool))
        idx += 1

    # ── FP-11: Identity Document Reuse ────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['IDENTITY_DOC_REUSE']} IDENTITY_DOC_REUSE records...")
    doc_generated = 0
    while doc_generated < FRAUD_COUNTS['IDENTITY_DOC_REUSE']:
        # Each shared PAN is reused across 2–4 records
        shared_pan = make_pan_hash(fake.bothify('REUSE????######'))
        batch_size = random.randint(2, 4)
        batch_size = min(batch_size, FRAUD_COUNTS['IDENTITY_DOC_REUSE'] - doc_generated)
        for _ in range(batch_size):
            records.append(generate_identity_doc_reuse(idx, referrer_pool, shared_pan))
            idx           += 1
            doc_generated += 1

    # ── FP-12: Referral Code Farming ─────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['REFERRAL_CODE_FARMING']} REFERRAL_CODE_FARMING records...")
    for _ in range(FRAUD_COUNTS['REFERRAL_CODE_FARMING']):
        records.append(generate_referral_code_farming(idx, referrer_pool))
        idx += 1

    # ── FP-10: No Usage Intent ────────────────────────────────────────────────
    print(f"Injecting {FRAUD_COUNTS['NO_USAGE_INTENT']} NO_USAGE_INTENT records...")
    for _ in range(FRAUD_COUNTS['NO_USAGE_INTENT']):
        records.append(generate_no_usage_intent(idx, referrer_pool))
        idx += 1

    df = pd.DataFrame(records)
    # Shuffle so fraud rows are not all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    total   = len(df)
    n_fraud = df['is_fraud'].sum()
    n_legit = total - n_fraud
    fraud_pct = n_fraud / total * 100

    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    print(f"Total records      : {total:,}")
    print(f"Legitimate records : {n_legit:,}  ({100-fraud_pct:.1f}%)")
    print(f"Fraud records      : {n_fraud:,}  ({fraud_pct:.1f}%)")
    print()

    print("Fraud type distribution:")
    dist = df[df['is_fraud'] == 1]['fraud_type'].value_counts()
    for ftype, count in dist.items():
        pct = count / n_fraud * 100
        print(f"  {ftype:<30} {count:>4}  ({pct:.1f}% of fraud)")

    print()
    print("Key column checks:")
    # R01: same device
    same_dev = (df['referrer_device_id'] == df['referee_device_id']).sum()
    print(f"  R01 same_device rows        : {same_dev}")
    # R03: disposable email
    disp = df['email_domain'].isin(DISPOSABLE_DOMAINS).sum()
    print(f"  R03 disposable_email rows   : {disp}")
    # R04: ip_account_count > 3
    ip_reuse = (df['ip_account_count'] > 3).sum()
    print(f"  R04 ip_account_count > 3    : {ip_reuse}")
    # R05: ring cluster
    ring_rows = (df['referral_cluster_size'] > 3).sum()
    print(f"  R05 cluster_size > 3        : {ring_rows}")
    # R08: identity_risk_score > 0.7
    synth = (df['identity_risk_score'] > 0.7).sum()
    print(f"  R08 identity_risk_score>0.7 : {synth}")
    # R09: burst velocity
    burst = ((df['referrals_in_1hr'] > 3) | (df['min_gap_sec'] < 120)).sum()
    print(f"  R09 burst velocity rows     : {burst}")
    # R10: geo velocity
    geo = (df['geo_velocity_kmph'] > 900).sum()
    print(f"  R10 geo_velocity > 900      : {geo}")
    # R12: no usage intent
    no_use = (~df['card_activated'] | (df['cash_advance_ratio'] > 0.8)).sum()
    print(f"  R12 no_usage_intent rows    : {no_use}")
    # R13: duplicate pan_hash
    dup_pan = df.duplicated(subset='pan_hash', keep=False).sum()
    print(f"  R13 duplicate pan_hash rows : {dup_pan}")
    # R15: code farming
    farming = (df['referral_code_usage_7d'] > 20).sum()
    print(f"  R15 code_usage > 20         : {farming}")

    print()
    print("Missing values per column:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  None — dataset is complete.")
    else:
        print(nulls)
    print("="*60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    print("Building synthetic referral fraud dataset...")
    print(f"Target: {TOTAL_RECORDS} records ({TOTAL_LEGIT} legit + {TOTAL_FRAUD} fraud)\n")

    df = build_dataset()
    validate(df)

    out_path = 'data/referral_dataset.csv'
    df.to_csv(out_path, index=False)
    print(f"\nDataset saved to: {out_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col} — dtype: {df[col].dtype}")
