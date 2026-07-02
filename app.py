"""
Network Anomaly Detection — Kunal's Lab
Flask web application for classifying network traffic patterns.

Run: python app.py
"""

import os
import json
import math
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
model = scaler = label_encoder = metadata = None


def load_models():
    global model, scaler, label_encoder, metadata
    paths = {
        'model':   os.path.join(MODELS_DIR, 'anomaly_model.joblib'),
        'scaler':  os.path.join(MODELS_DIR, 'scaler.joblib'),
        'encoder': os.path.join(MODELS_DIR, 'label_encoder.joblib'),
        'meta':    os.path.join(MODELS_DIR, 'metadata.json'),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        raise RuntimeError("Models not found. Run `python train.py` first.")
    model         = joblib.load(paths['model'])
    scaler        = joblib.load(paths['scaler'])
    label_encoder = joblib.load(paths['encoder'])
    with open(paths['meta']) as f:
        metadata = json.load(f)


def preprocess_input(form):
    duration       = float(form.get('duration', 30))
    bytes_sent     = float(form.get('bytes_sent', 5000))
    bytes_recv     = float(form.get('bytes_recv', 8000))
    packets_sent   = int(form.get('packets_sent', 50))
    packets_recv   = int(form.get('packets_recv', 60))
    src_port       = int(form.get('src_port', 54321))
    dst_port       = int(form.get('dst_port', 443))
    protocol       = int(form.get('protocol', 6))
    unique_ips     = int(form.get('unique_ips', 2))
    failed_logins  = int(form.get('failed_logins', 0))
    syn_count      = int(form.get('syn_count', 3))
    fin_count      = int(form.get('fin_count', 3))
    rst_count      = int(form.get('rst_count', 1))
    ttl_mean       = float(form.get('ttl_mean', 64))
    flow_iat_mean  = float(form.get('flow_iat_mean', 0.5))

    # Engineered features
    bytes_ratio     = math.log1p(bytes_recv) / (math.log1p(bytes_sent) + 1e-6)
    packet_ratio    = packets_recv / (packets_sent + 1)
    bytes_per_pkt   = math.log1p(bytes_sent + bytes_recv) / (packets_sent + packets_recv + 1)
    syn_rst_ratio   = syn_count / (rst_count + 1)
    is_well_known   = int(dst_port < 1024)
    is_encrypted    = int(dst_port in [443, 8443, 993, 995])
    duration_log    = math.log1p(duration)
    flow_iat_log    = math.log1p(flow_iat_mean)

    feat_dict = {
        'duration': duration, 'bytes_sent': bytes_sent, 'bytes_recv': bytes_recv,
        'packets_sent': packets_sent, 'packets_recv': packets_recv,
        'src_port': src_port, 'dst_port': dst_port, 'protocol': protocol,
        'unique_ips': unique_ips, 'failed_logins': failed_logins,
        'syn_count': syn_count, 'fin_count': fin_count, 'rst_count': rst_count,
        'ttl_mean': ttl_mean, 'flow_iat_mean': flow_iat_mean,
        'bytes_ratio': bytes_ratio, 'packet_ratio': packet_ratio,
        'bytes_per_pkt': bytes_per_pkt, 'syn_rst_ratio': syn_rst_ratio,
        'is_well_known_dst': is_well_known, 'is_encrypted': is_encrypted,
        'duration_log': duration_log, 'flow_iat_log': flow_iat_log,
    }

    feature_cols = metadata['feature_columns']
    row = np.array([[feat_dict.get(col, 0.0) for col in feature_cols]])
    return row


SEVERITY = {
    'Normal': 'none',
    'Port Scan': 'medium',
    'Brute Force': 'high',
    'Data Exfiltration': 'critical',
    'DoS Attack': 'critical',
}

ACTIONS = {
    'Normal': 'No action required — traffic is within normal parameters.',
    'Port Scan': 'Flag source IP. Review firewall rules and close unused ports.',
    'Brute Force': 'Block source IP immediately. Force credential reset on target accounts.',
    'Data Exfiltration': 'CRITICAL — isolate affected host, alert security team, preserve logs.',
    'DoS Attack': 'CRITICAL — engage rate limiting and upstream null-routing. Alert NOC.',
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        row    = preprocess_input(request.form)
        scaled = scaler.transform(row)
        pred   = int(model.predict(scaled)[0])
        proba  = model.predict_proba(scaled)[0]

        label      = label_encoder.inverse_transform([pred])[0]
        confidence = round(float(proba[pred]) * 100, 1)
        classes    = label_encoder.classes_.tolist()

        class_probs = {cls: round(float(p) * 100, 1)
                       for cls, p in zip(classes, proba)}

        return jsonify({
            'prediction':  label,
            'confidence':  confidence,
            'severity':    SEVERITY.get(label, 'medium'),
            'action':      ACTIONS.get(label, ''),
            'class_probs': class_probs,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'metrics': metadata})


if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5002)
