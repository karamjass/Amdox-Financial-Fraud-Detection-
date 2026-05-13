"""
FraudShield — Model Training (PaySim Dataset)
==============================================
Trains RandomForest on PaySim1 with balanced undersampling.
~6.3M transactions → balanced to equal fraud/legit samples.

Features (14): step, amount, balances, derived, one-hot type
Input : data/PS_20174392719_1491204439457_log.csv
Output: trained_models/fraud_model.joblib + scaler + metadata
Run   : python train.py
"""

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix
)
import joblib

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'PS_20174392719_1491204439457_log.csv')
MODEL_PATH   = os.path.join(MODEL_DIR, 'fraud_model.joblib')
SCALER_PATH  = os.path.join(MODEL_DIR, 'fraud_scaler.joblib')
META_PATH    = os.path.join(MODEL_DIR, 'training_metadata.json')

TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

FEATURE_COLS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_diff_orig', 'balance_diff_dest', 'orig_balance_ratio',
] + [f'type_{t}' for t in TRANSACTION_TYPES]

FEATURE_GROUPS = {
    'step': 'Transaction', 'amount': 'Transaction',
    'type_CASH_IN': 'Transaction', 'type_CASH_OUT': 'Transaction',
    'type_DEBIT': 'Transaction', 'type_PAYMENT': 'Transaction',
    'type_TRANSFER': 'Transaction',
    'oldbalanceOrg': 'Sender', 'newbalanceOrig': 'Sender',
    'balance_diff_orig': 'Sender',
    'oldbalanceDest': 'Receiver', 'newbalanceDest': 'Receiver',
    'balance_diff_dest': 'Receiver',
    'orig_balance_ratio': 'Derived',
}

TARGET = 'isFraud'


def header(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")


def engineer_features(df):
    df = df.copy()
    df['balance_diff_orig']  = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_dest']  = df['newbalanceDest'] - df['oldbalanceDest']
    df['orig_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    for t in TRANSACTION_TYPES:
        df[f'type_{t}'] = (df['type'] == t).astype(int)
    return df


def main():
    header("FRAUDSHIELD — PAYSIM DATASET TRAINING")

    if not os.path.exists(DATASET_PATH):
        print(f"\n  ❌ Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    print(f"\n  Loading: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Total rows : {len(df):,}")
    print(f"  Fraud      : {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)")
    print(f"  Legit      : {(~df[TARGET].astype(bool)).sum():,}")

    # ─── BALANCED UNDERSAMPLING (user-specified) ───
    fraud = df[df['isFraud'] == 1]
    legit = df[df['isFraud'] == 0].sample(len(fraud), random_state=42)
    df_balanced = pd.concat([fraud, legit])
    print(f"\n  ─── BALANCED ───")
    print(f"  Fraud   : {len(fraud):,}")
    print(f"  Legit   : {len(legit):,}")
    print(f"  Total   : {len(df_balanced):,}")

    df_balanced = engineer_features(df_balanced)
    X = df_balanced[FEATURE_COLS]
    y = df_balanced[TARGET]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train : {len(X_train):,}  (fraud: {y_train.sum():,})")
    print(f"  Test  : {len(X_test):,}   (fraud: {y_test.sum():,})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\n  Training RandomForest (200 trees, max_depth=18)...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200, max_depth=18,
        min_samples_split=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    print(f"  ✅ Trained in {train_time:.1f}s")

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n  ─── RESULTS ───")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"\n  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    print()
    print(classification_report(y_test, y_pred,
          target_names=['Legit', 'Fraud'], zero_division=0))

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("  ─── FEATURE IMPORTANCES ───")
    for feat, imp in sorted_imp:
        grp = FEATURE_GROUPS.get(feat, '?')
        bar = '█' * int(imp * 80)
        print(f"  [{grp:12s}] {feat:22s} {imp:.4f}  {bar}")

    group_scores = {}
    for feat, imp in importances.items():
        grp = FEATURE_GROUPS.get(feat, 'Other')
        group_scores[grp] = group_scores.get(grp, 0) + imp
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  ─── GROUP IMPORTANCES ───")
    for grp, score in sorted_groups:
        bar = '█' * int(score * 30)
        print(f"  {grp:15s} {score:.4f}  {bar}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  ✅ Model  : {MODEL_PATH}")
    print(f"  ✅ Scaler : {SCALER_PATH}")

    meta = {
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_time_seconds': round(train_time, 2),
        'dataset': 'PaySim1 (Kaggle Synthetic Financial Dataset)',
        'dataset_source': 'Kaggle ealaxi/paysim1',
        'total_samples_original': len(df),
        'total_samples_balanced': len(df_balanced),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_fraud': int(df[TARGET].sum()),
        'fraud_rate_pct': round(df[TARGET].mean() * 100, 3),
        'balancing_method': 'undersampling',
        'features': FEATURE_COLS,
        'feature_groups': FEATURE_GROUPS,
        'transaction_types': TRANSACTION_TYPES,
        'model_type': 'RandomForestClassifier',
        'model_params': {'n_estimators': 200, 'max_depth': 18},
        'metrics': {
            'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
        },
        'confusion_matrix': {
            'true_negative': int(cm[0][0]), 'false_positive': int(cm[0][1]),
            'false_negative': int(cm[1][0]), 'true_positive': int(cm[1][1]),
        },
        'feature_importances': {k: round(v, 4) for k, v in sorted_imp},
        'group_importances': {k: round(v, 4) for k, v in sorted_groups},
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Meta   : {META_PATH}")

    header(f"DONE — {train_time:.1f}s  |  Acc {acc:.4f}  |  AUC {auc:.4f}")
    print(f"  Next: python app.py\n")


if __name__ == '__main__':
    main()
