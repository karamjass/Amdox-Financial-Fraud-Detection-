"""
FraudShield — Flask Backend (PaySim + Email Phishing)
=====================================================
Endpoints:
  GET  /              → Dashboard
  POST /predict       → Fraud prediction (PaySim)
  GET  /api/stats     → Dashboard stats + model metrics
  GET  /api/recent    → Recent prediction logs
  GET  /api/dataset   → Sample rows from PaySim
  GET  /api/random_transaction → Random txn for analyzer
  POST /api/analyze_email → Email phishing via Groq LLM
  POST /api/clear     → Clear logs

Run: python app.py → http://127.0.0.1:5000
"""

import os, json, time, base64, re
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ─── GROQ API KEY (paste your key here) ───
GROQ_API_KEY = "[ENCRYPTION_KEY]"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'trained_models')
LOGS_DIR   = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE     = os.path.join(LOGS_DIR, 'predictions.json')
MODEL_PATH   = os.path.join(MODEL_DIR, 'fraud_model.joblib')
SCALER_PATH  = os.path.join(MODEL_DIR, 'fraud_scaler.joblib')
META_PATH    = os.path.join(MODEL_DIR, 'training_metadata.json')
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'PS_20174392719_1491204439457_log.csv')

model  = None
scaler = None
meta   = {}

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


def load_model():
    global model, scaler, meta
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("  [OK] ML model loaded (PaySim balanced)")
    else:
        print("  [WARNING] No model found. Run `python train.py` first.")
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            meta = json.load(f)


def load_logs():
    if not os.path.exists(LOG_FILE): return []
    try:
        with open(LOG_FILE, 'r') as f: return json.load(f)
    except: return []


def save_logs(logs):
    with open(LOG_FILE, 'w') as f:
        json.dump(logs[-500:], f, indent=2)


def build_explanation(raw, fraud_prob):
    reasons = []
    amt = raw.get('amount', 0)
    txn_type = raw.get('type', '')
    old_o = raw.get('oldbalanceOrg', 0)
    new_o = raw.get('newbalanceOrig', 0)
    old_d = raw.get('oldbalanceDest', 0)
    new_d = raw.get('newbalanceDest', 0)

    if amt > 200000:
        reasons.append({'type': 'Transaction', 'icon': '💰',
            'text': f'Very high amount (${amt:,.2f})', 'severity': 'high'})
    elif amt > 50000:
        reasons.append({'type': 'Transaction', 'icon': '💰',
            'text': f'Large amount (${amt:,.2f})', 'severity': 'medium'})

    if txn_type in ['TRANSFER', 'CASH_OUT']:
        reasons.append({'type': 'Transaction', 'icon': '💸',
            'text': f'High-risk type: {txn_type}', 'severity': 'medium'})

    if old_o > 0 and new_o == 0:
        reasons.append({'type': 'Sender', 'icon': '🏦',
            'text': f'Sender fully drained (${old_o:,.2f} → $0)', 'severity': 'high'})
    elif amt > old_o and old_o > 0:
        reasons.append({'type': 'Sender', 'icon': '⚠️',
            'text': f'Amount exceeds sender balance', 'severity': 'high'})

    if old_d == 0 and new_d == 0 and amt > 0:
        reasons.append({'type': 'Receiver', 'icon': '👤',
            'text': 'Receiver balance unchanged despite transfer', 'severity': 'high'})

    if old_o > 0:
        ratio = amt / old_o
        if ratio > 0.9:
            reasons.append({'type': 'Derived', 'icon': '📊',
                'text': f'Transaction is {ratio:.0%} of sender balance', 'severity': 'high' if ratio > 0.95 else 'medium'})

    if fraud_prob > 0.80:
        reasons.append({'type': 'Model', 'icon': '🚨',
            'text': f'Extreme fraud probability ({fraud_prob:.1%})', 'severity': 'high'})
    elif fraud_prob > 0.50:
        reasons.append({'type': 'Model', 'icon': '⚠️',
            'text': f'Elevated fraud probability ({fraud_prob:.1%})', 'severity': 'medium'})

    if not reasons:
        reasons.append({'type': 'Overall', 'icon': '✅',
            'text': 'No suspicious indicators — appears normal', 'severity': 'safe'})
    return reasons


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON body'}), 400
        now = datetime.now()

        raw = {
            'step': float(data.get('step', 1)),
            'type': data.get('type', 'TRANSFER'),
            'amount': float(data.get('amount', 0)),
            'oldbalanceOrg': float(data.get('oldbalanceOrg', 0)),
            'newbalanceOrig': float(data.get('newbalanceOrig', 0)),
            'oldbalanceDest': float(data.get('oldbalanceDest', 0)),
            'newbalanceDest': float(data.get('newbalanceDest', 0)),
        }

        features = {
            'step': raw['step'], 'amount': raw['amount'],
            'oldbalanceOrg': raw['oldbalanceOrg'],
            'newbalanceOrig': raw['newbalanceOrig'],
            'oldbalanceDest': raw['oldbalanceDest'],
            'newbalanceDest': raw['newbalanceDest'],
            'balance_diff_orig': raw['oldbalanceOrg'] - raw['newbalanceOrig'],
            'balance_diff_dest': raw['newbalanceDest'] - raw['oldbalanceDest'],
            'orig_balance_ratio': raw['amount'] / (raw['oldbalanceOrg'] + 1),
        }
        for t in TRANSACTION_TYPES:
            features[f'type_{t}'] = 1 if raw['type'] == t else 0

        if model is not None and scaler is not None:
            X = np.array([[features[c] for c in FEATURE_COLS]])
            X_s = scaler.transform(X)
            fraud_prob = float(model.predict_proba(X_s)[0][1])
            is_fraud   = bool(model.predict(X_s)[0])
            method     = 'ml_model'
        else:
            fraud_prob = 0.0
            if raw['amount'] > 200000: fraud_prob += 0.2
            if raw['type'] in ['TRANSFER', 'CASH_OUT']: fraud_prob += 0.15
            if raw['oldbalanceOrg'] > 0 and raw['newbalanceOrig'] == 0: fraud_prob += 0.3
            is_fraud = fraud_prob > 0.5
            method   = 'rule_based'

        if fraud_prob > 0.80:
            severity = 'CRITICAL'
            rec = '🚨 BLOCK immediately — extremely high fraud probability.'
        elif fraud_prob > 0.55:
            severity = 'HIGH'
            rec = '⚠️ FLAG for manual review — suspicious patterns.'
        elif fraud_prob > 0.30:
            severity = 'MEDIUM'
            rec = '⚡ MONITOR — some risk indicators present.'
        else:
            severity = 'LOW'
            rec = '✅ Transaction appears safe.'

        reasons = build_explanation(raw, fraud_prob)

        result = {
            'success': True, 'fraud_probability': round(fraud_prob, 4),
            'is_fraud': is_fraud, 'severity': severity,
            'recommendation': rec, 'method': method,
            'reasons': reasons, 'timestamp': now.isoformat(),
        }

        logs = load_logs()
        logs.append({
            'id': len(logs) + 1, 'type': raw['type'],
            'amount': raw['amount'],
            'fraud_probability': round(fraud_prob, 4),
            'is_fraud': is_fraud, 'severity': severity,
            'method': method,
            'reasons_count': len([r for r in reasons if r['severity'] != 'safe']),
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        })
        save_logs(logs)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    logs = load_logs()
    total = len(logs)
    fraud_cnt = sum(1 for l in logs if l.get('is_fraud'))
    metrics = meta.get('metrics', {})
    return jsonify({
        'total_predictions': total,
        'fraud_detected': fraud_cnt,
        'safe_transactions': total - fraud_cnt,
        'model_loaded': model is not None,
        'model_type': meta.get('model_type', 'N/A'),
        'dataset': meta.get('dataset', 'N/A'),
        'dataset_source': meta.get('dataset_source', 'N/A'),
        'total_dataset_rows': meta.get('total_samples_original', 0),
        'total_dataset_fraud': meta.get('total_fraud', 0),
        'total_balanced': meta.get('total_samples_balanced', 0),
        'fraud_rate_pct': meta.get('fraud_rate_pct', 0),
        'balancing_method': meta.get('balancing_method', 'N/A'),
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1_score': metrics.get('f1_score', 0),
        'roc_auc': metrics.get('roc_auc', 0),
        'confusion_matrix': meta.get('confusion_matrix', {}),
        'feature_importances': meta.get('feature_importances', {}),
        'group_importances': meta.get('group_importances', {}),
        'feature_groups': FEATURE_GROUPS,
    })


@app.route('/api/recent')
def get_recent():
    return jsonify(load_logs()[-50:][::-1])


@app.route('/api/dataset')
def get_dataset():
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        df = pd.read_csv(DATASET_PATH, nrows=500000)
        fraud_rows = df[df['isFraud'] == 1].sample(n=min(50, df['isFraud'].sum()), random_state=42)
        legit_rows = df[df['isFraud'] == 0].sample(n=50, random_state=42)
        sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42)
        records = []
        for _, row in sample.iterrows():
            records.append({
                'step': int(row['step']), 'type': row['type'],
                'amount': round(float(row['amount']), 2),
                'oldbalanceOrg': round(float(row['oldbalanceOrg']), 2),
                'newbalanceOrig': round(float(row['newbalanceOrig']), 2),
                'oldbalanceDest': round(float(row['oldbalanceDest']), 2),
                'newbalanceDest': round(float(row['newbalanceDest']), 2),
                'isFraud': int(row['isFraud']),
            })
        total_fraud = int(df['isFraud'].sum())
        return jsonify({
            'success': True, 'total_rows': len(df),
            'total_fraud': total_fraud, 'total_legit': len(df) - total_fraud,
            'fraud_pct': round(total_fraud / len(df) * 100, 3),
            'sample': records,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/random_transaction')
def random_transaction():
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({'success': False, 'error': 'Dataset not found'}), 404
        df = pd.read_csv(DATASET_PATH, nrows=500000)
        want = request.args.get('fraud', 'random')
        if want == '1':
            row = df[df['isFraud'] == 1].sample(1).iloc[0]
        elif want == '0':
            row = df[df['isFraud'] == 0].sample(1).iloc[0]
        else:
            row = df.sample(1).iloc[0]
        rec = {
            'step': int(row['step']), 'type': row['type'],
            'amount': round(float(row['amount']), 4),
            'oldbalanceOrg': round(float(row['oldbalanceOrg']), 4),
            'newbalanceOrig': round(float(row['newbalanceOrig']), 4),
            'oldbalanceDest': round(float(row['oldbalanceDest']), 4),
            'newbalanceDest': round(float(row['newbalanceDest']), 4),
            'actual_label': int(row['isFraud']),
        }
        return jsonify({'success': True, 'transaction': rec})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze_email', methods=['POST'])
def analyze_email():
    """Analyze email for phishing using Groq LLM."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data'}), 400

        email_text  = data.get('text', '')
        email_image = data.get('image', '')
        api_key     = GROQ_API_KEY if GROQ_API_KEY != 'YOUR_GROQ_API_KEY_HERE' else os.environ.get('GROQ_API_KEY', '')

        if not api_key:
            return jsonify({'success': False, 'error': 'Set your Groq API key in app.py (GROQ_API_KEY variable) or as GROQ_API_KEY env var.'}), 400
        if not email_text and not email_image:
            return jsonify({'success': False, 'error': 'Provide email text or image'}), 400

        try:
            from groq import Groq
        except ImportError:
            return jsonify({'success': False, 'error': 'Run: pip install groq'}), 500

        client = Groq(api_key=api_key)

        prompt = """You are an expert email security analyst. Analyze the following email for phishing indicators.

Score EACH factor individually out of 100. The overall score is the weighted average.

You MUST respond with ONLY valid JSON (no markdown, no code blocks) in this exact format:
{
    "score": <integer 1-100, overall phishing risk>,
    "verdict": "<PHISHING or SUSPICIOUS or SAFE>",
    "summary": "<one sentence summary>",
    "factor_scores": {
        "urgency": {"score": <0-100>, "reason": "<why this score>"},
        "sender_legitimacy": {"score": <0-100>, "reason": "<why this score>"},
        "link_safety": {"score": <0-100>, "reason": "<why this score>"},
        "grammar_quality": {"score": <0-100>, "reason": "<why this score>"},
        "impersonation": {"score": <0-100>, "reason": "<why this score>"},
        "data_request": {"score": <0-100>, "reason": "<why this score>"},
        "attachment_risk": {"score": <0-100>, "reason": "<why this score>"},
        "emotional_manipulation": {"score": <0-100>, "reason": "<why this score>"}
    },
    "indicators": [
        {"text": "<specific phishing red flag found>", "severity": "<high or medium or low>", "category": "<urgency|sender|link|grammar|impersonation|attachment|request|other>", "score": <factor score 0-100>}
    ]
}

Rules:
- Score each factor independently from 0 (no risk) to 100 (extreme risk)
- The overall score should reflect the combined risk across ALL factors
- If a factor is not applicable (e.g. no links), score it 0
- List ALL specific red flags as indicators with their individual scores
- If email is safe, all factor scores should be low and indicators array minimal"""

        messages = [{"role": "system", "content": prompt}]

        if email_image:
            # Vision model for images
            image_b64 = email_image.split(',')[-1] if ',' in email_image else email_image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this email screenshot for phishing:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            })
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages, temperature=0.3, max_tokens=2000,
            )
        else:
            messages.append({"role": "user", "content": f"EMAIL CONTENT:\n{email_text}"})
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b", messages=messages,
                temperature=0.3, max_tokens=2000,
            )

        response_text = resp.choices[0].message.content.strip()
        # Clean markdown wrappers
        if '```' in response_text:
            response_text = re.sub(r'```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```', '', response_text)
        # Find JSON in response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            response_text = match.group()

        result = json.loads(response_text)
        result['success'] = True
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            'success': True, 'score': 50, 'verdict': 'SUSPICIOUS',
            'summary': 'Analysis completed but format was unexpected.',
            'indicators': [{'text': response_text[:300], 'severity': 'medium', 'category': 'other'}],
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_logs_route():
    try:
        save_logs([])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


load_model()

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  FraudShield -- PaySim + Email Phishing Analyzer")
    print("=" * 60)
    print(f"  Dashboard : http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
