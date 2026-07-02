"""
Network Anomaly Detection — Kunal's Lab
Multi-class classification of network traffic into:
  Normal, DoS Attack, Port Scan, Brute Force, Data Exfiltration

Run: python train.py
Outputs: models/anomaly_model.joblib, models/scaler.joblib, models/metadata.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CLASSES = ['Normal', 'DoS Attack', 'Port Scan', 'Brute Force', 'Data Exfiltration']


# ── Synthetic network traffic generation ──────────────────────────────────────
def generate_traffic(n=60000, random_state=42):
    np.random.seed(random_state)
    rows = []

    profiles = {
        'Normal': dict(
            duration      = lambda n: np.random.exponential(30, n),
            bytes_sent    = lambda n: np.random.lognormal(8, 1.5, n),
            bytes_recv    = lambda n: np.random.lognormal(9, 1.5, n),
            packets_sent  = lambda n: np.random.poisson(50, n),
            packets_recv  = lambda n: np.random.poisson(60, n),
            src_port      = lambda n: np.random.choice([80,443,8080,3000,22,3306], n),
            dst_port      = lambda n: np.random.choice([80,443,8080,3000,22,3306], n),
            protocol      = lambda n: np.random.choice([6,17,1], n, p=[0.7,0.2,0.1]),
            unique_ips    = lambda n: np.random.randint(1, 5, n),
            failed_logins = lambda n: np.zeros(n, dtype=int),
            syn_count     = lambda n: np.random.poisson(3, n),
            fin_count     = lambda n: np.random.poisson(3, n),
            rst_count     = lambda n: np.random.poisson(1, n),
            ttl_mean      = lambda n: np.random.normal(64, 5, n),
            flow_iat_mean = lambda n: np.random.exponential(0.5, n),
            label         = 'Normal'
        ),
        'DoS Attack': dict(
            duration      = lambda n: np.random.exponential(5, n),
            bytes_sent    = lambda n: np.random.lognormal(11, 1, n),   # huge volume
            bytes_recv    = lambda n: np.random.lognormal(5, 0.5, n),
            packets_sent  = lambda n: np.random.poisson(5000, n),
            packets_recv  = lambda n: np.random.poisson(20, n),
            src_port      = lambda n: np.random.randint(1024, 65535, n),
            dst_port      = lambda n: np.full(n, 80),
            protocol      = lambda n: np.random.choice([6,17,1], n, p=[0.5,0.3,0.2]),
            unique_ips    = lambda n: np.random.randint(1, 3, n),
            failed_logins = lambda n: np.zeros(n, dtype=int),
            syn_count     = lambda n: np.random.poisson(500, n),       # SYN flood
            fin_count     = lambda n: np.zeros(n, dtype=int),
            rst_count     = lambda n: np.random.poisson(200, n),
            ttl_mean      = lambda n: np.random.normal(128, 2, n),
            flow_iat_mean = lambda n: np.random.exponential(0.001, n), # very fast
            label         = 'DoS Attack'
        ),
        'Port Scan': dict(
            duration      = lambda n: np.random.exponential(0.5, n),
            bytes_sent    = lambda n: np.random.lognormal(4, 0.5, n),
            bytes_recv    = lambda n: np.random.lognormal(4, 0.5, n),
            packets_sent  = lambda n: np.random.poisson(2, n),
            packets_recv  = lambda n: np.random.poisson(1, n),
            src_port      = lambda n: np.random.randint(1024, 65535, n),
            dst_port      = lambda n: np.random.randint(1, 65535, n),  # scanning all ports
            protocol      = lambda n: np.full(n, 6),
            unique_ips    = lambda n: np.random.randint(1, 3, n),
            failed_logins = lambda n: np.zeros(n, dtype=int),
            syn_count     = lambda n: np.random.poisson(50, n),
            fin_count     = lambda n: np.random.poisson(1, n),
            rst_count     = lambda n: np.random.poisson(50, n),        # many resets
            ttl_mean      = lambda n: np.random.normal(64, 3, n),
            flow_iat_mean = lambda n: np.random.exponential(0.01, n),
            label         = 'Port Scan'
        ),
        'Brute Force': dict(
            duration      = lambda n: np.random.exponential(60, n),
            bytes_sent    = lambda n: np.random.lognormal(6, 0.8, n),
            bytes_recv    = lambda n: np.random.lognormal(6, 0.8, n),
            packets_sent  = lambda n: np.random.poisson(100, n),
            packets_recv  = lambda n: np.random.poisson(100, n),
            src_port      = lambda n: np.random.randint(1024, 65535, n),
            dst_port      = lambda n: np.random.choice([22, 3389, 21, 23], n),
            protocol      = lambda n: np.full(n, 6),
            unique_ips    = lambda n: np.random.randint(1, 4, n),
            failed_logins = lambda n: np.random.poisson(30, n),        # many failures
            syn_count     = lambda n: np.random.poisson(30, n),
            fin_count     = lambda n: np.random.poisson(30, n),
            rst_count     = lambda n: np.random.poisson(5, n),
            ttl_mean      = lambda n: np.random.normal(64, 4, n),
            flow_iat_mean = lambda n: np.random.exponential(1.0, n),
            label         = 'Brute Force'
        ),
        'Data Exfiltration': dict(
            duration      = lambda n: np.random.exponential(120, n),
            bytes_sent    = lambda n: np.random.lognormal(5, 0.5, n),
            bytes_recv    = lambda n: np.random.lognormal(11, 1.2, n), # huge download
            packets_sent  = lambda n: np.random.poisson(20, n),
            packets_recv  = lambda n: np.random.poisson(1000, n),
            src_port      = lambda n: np.random.randint(1024, 65535, n),
            dst_port      = lambda n: np.random.choice([443, 80, 53, 8443], n),
            protocol      = lambda n: np.random.choice([6,17], n, p=[0.6,0.4]),
            unique_ips    = lambda n: np.random.randint(1, 3, n),
            failed_logins = lambda n: np.zeros(n, dtype=int),
            syn_count     = lambda n: np.random.poisson(5, n),
            fin_count     = lambda n: np.random.poisson(5, n),
            rst_count     = lambda n: np.random.poisson(1, n),
            ttl_mean      = lambda n: np.random.normal(64, 5, n),
            flow_iat_mean = lambda n: np.random.exponential(2.0, n),
            label         = 'Data Exfiltration'
        ),
    }

    counts = {
        'Normal': int(n * 0.60), 'DoS Attack': int(n * 0.15),
        'Port Scan': int(n * 0.10), 'Brute Force': int(n * 0.10),
        'Data Exfiltration': int(n * 0.05)
    }

    dfs = []
    for cls, prof in profiles.items():
        cnt = counts[cls]
        row = {k: v(cnt) for k, v in prof.items() if k != 'label'}
        row['label'] = cls
        dfs.append(pd.DataFrame(row))

    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    df['bytes_ratio']   = np.log1p(df['bytes_recv']) / (np.log1p(df['bytes_sent']) + 1e-6)
    df['packet_ratio']  = df['packets_recv'] / (df['packets_sent'] + 1)
    df['bytes_per_pkt'] = np.log1p(df['bytes_sent'] + df['bytes_recv']) / (df['packets_sent'] + df['packets_recv'] + 1)
    df['syn_rst_ratio'] = df['syn_count'] / (df['rst_count'] + 1)
    df['is_well_known_dst'] = (df['dst_port'] < 1024).astype(int)
    df['is_encrypted']      = df['dst_port'].isin([443, 8443, 993, 995]).astype(int)
    df['duration_log']      = np.log1p(df['duration'])
    df['flow_iat_log']      = np.log1p(df['flow_iat_mean'])
    return df


def get_feature_cols(df):
    return [c for c in df.columns if c != 'label']


def train():
    print("Generating synthetic network traffic dataset...")
    df = generate_traffic()
    print(f"  Total samples: {len(df):,}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")
    df.to_csv(os.path.join(DATA_DIR, 'network_traffic.csv'), index=False)

    df = engineer_features(df)
    feature_cols = get_feature_cols(df)

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(df['label'])
    X = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("\nTraining ensemble (Random Forest + Extra Trees)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=14,
                                 min_samples_leaf=2, n_jobs=-1, random_state=42)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=14,
                               min_samples_leaf=2, n_jobs=-1, random_state=42)
    clf = VotingClassifier(estimators=[('rf', rf), ('et', et)],
                           voting='soft', weights=[2, 1])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*40}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"{'─'*40}")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_))

    metadata = {
        'accuracy':       round(acc, 4),
        'classes':        le.classes_.tolist(),
        'feature_columns': feature_cols,
    }

    joblib.dump(clf,    os.path.join(MODELS_DIR, 'anomaly_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(le,     os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Models saved to models/")


if __name__ == '__main__':
    train()
