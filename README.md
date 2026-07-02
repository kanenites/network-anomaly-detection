# Network Anomaly Detection

Multi-class network traffic classifier that detects five traffic types — Normal, DoS Attack, Port Scan, Brute Force, and Data Exfiltration — using a soft-voting ensemble of Random Forest and Extra Trees classifiers, served via a Flask web app.

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 100% |
| Test Accuracy | 100% |
| CV Accuracy (5-fold) | 1.000 ± 0.000 |

> Scores reflect performance on synthetic data with well-separated class distributions per attack type. For production use, replace `train.py`'s generator with real capture data such as [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) or [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html).

## Detection Classes

| Class | Description | Severity |
|---|---|---|
| Normal | Legitimate traffic | None |
| Port Scan | Sequential port probing | Medium |
| Brute Force | Repeated auth attempts on SSH/RDP/FTP | High |
| DoS Attack | High-volume SYN/UDP flood | Critical |
| Data Exfiltration | Large outbound transfer to external hosts | Critical |

## Intended Audience

This tool is designed for **network security engineers and SOC analysts**, not end users. The input parameters (SYN/FIN/RST counts, Flow IAT, TTL) are not values a typical user would know — they come from network monitoring tools such as:

- **Wireshark / tshark** — packet capture and flow analysis
- **tcpdump** — command-line packet analyser
- **CICFlowMeter** — generates flow-level features from PCAPs
- **SIEM platforms** (Splunk, Elastic SIEM) — aggregate these metrics automatically

In a production deployment, this model would sit behind an automated pipeline that extracts these features from live traffic and calls the `/predict` endpoint directly — no manual input required.



1. Network flow parameters are submitted (bytes, packets, ports, protocol, TCP flags, TTL, IAT)
2. Eight features are engineered from raw inputs (see below)
3. Random Forest + Extra Trees ensemble classifies the traffic type
4. The app returns per-class probabilities, severity level, and a recommended action

## Tech Stack

Python · scikit-learn · Flask · NumPy · Pandas

## Model Details

**Algorithm:** Soft-voting ensemble — `RandomForestClassifier` (weight 2) + `ExtraTreesClassifier` (weight 1)

**Raw features (15):** `duration`, `bytes_sent`, `bytes_recv`, `packets_sent`, `packets_recv`, `src_port`, `dst_port`, `protocol`, `unique_ips`, `failed_logins`, `syn_count`, `fin_count`, `rst_count`, `ttl_mean`, `flow_iat_mean`

**Engineered features (8):**

| Feature | Formula | Captures |
|---|---|---|
| `bytes_ratio` | `log(bytes_recv) / log(bytes_sent)` | Exfiltration asymmetry |
| `packet_ratio` | `packets_recv / packets_sent` | Traffic directionality |
| `bytes_per_pkt` | `log(total_bytes) / total_packets` | Packet size patterns |
| `syn_rst_ratio` | `syn_count / (rst_count + 1)` | SYN flood indicator |
| `is_well_known_dst` | `dst_port < 1024` | Targeting common services |
| `is_encrypted` | `dst_port in [443, 8443, 993, 995]` | Encrypted channel use |
| `duration_log` | `log1p(duration)` | Normalise skewed duration |
| `flow_iat_log` | `log1p(flow_iat_mean)` | Normalise inter-arrival time |

## Project Structure

```
network-anomaly-detection/
├── train.py              # Traffic generation + model training
├── app.py                # Flask web app + /predict endpoint
├── templates/
│   └── index.html        # Kunal's Lab branded UI with probability bars
├── models/               # Saved artifacts (populated after training)
│   ├── anomaly_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── metadata.json
├── data/                 # Generated CSV (populated after training)
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/kanenites/network-anomaly-detection.git
cd network-anomaly-detection
pip install -r requirements.txt

# Train
python train.py

# Serve
python app.py
# → http://localhost:5002
```

---
*Kunal's Lab · AI & ML Systems*
