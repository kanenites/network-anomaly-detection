# Network Anomaly Detection

A multi-class network traffic classifier that detects five types of network behaviour — Normal, DoS Attack, Port Scan, Brute Force, and Data Exfiltration — using an ensemble ML model. Served via a branded Flask web app.

## What It Does

- Accepts network flow parameters (bytes, packets, ports, protocol, TCP flags, TTL, etc.)
- Classifies the traffic into one of five categories
- Returns confidence scores for each class, severity level, and a recommended security action

## Detection Classes

| Class | Description |
|---|---|
| Normal | Legitimate network traffic |
| DoS Attack | High-volume flood targeting a service |
| Port Scan | Sequential scanning of ports to find open services |
| Brute Force | Repeated authentication attempts against SSH/RDP/FTP |
| Data Exfiltration | Large outbound data transfer to external hosts |

## Tech Stack

Python · scikit-learn · Flask · NumPy · Pandas

## Project Structure

```
network-anomaly-detection/
├── train.py          # Traffic generation + model training
├── app.py            # Flask web application
├── templates/
│   └── index.html    # Branded UI with per-class probability bars
├── models/           # Saved model artifacts (generated at runtime)
├── data/             # Generated dataset (generated at runtime)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/kanenites/network-anomaly-detection.git
cd network-anomaly-detection
pip install -r requirements.txt
```

## Usage

**Step 1 — Train the model:**
```bash
python train.py
```
Generates 60,000 synthetic network flows across 5 classes, trains a Random Forest + Extra Trees ensemble, and saves the model to `models/`.

**Step 2 — Run the web app:**
```bash
python app.py
```
Open `http://localhost:5002` and enter network flow parameters to classify the traffic.

## Model Details

- **Algorithm:** Soft-voting ensemble (Random Forest + Extra Trees)
- **Features:** Duration, bytes sent/recv, packet counts, ports, protocol, TCP flags (SYN/FIN/RST), TTL, IAT, and engineered ratios (bytes ratio, packet ratio, SYN-RST ratio)
- **Engineered features:** `bytes_ratio`, `packet_ratio`, `bytes_per_pkt`, `syn_rst_ratio`, `is_well_known_dst`, `is_encrypted`, `duration_log`, `flow_iat_log`

> **Note:** Trained on synthetic data for demonstration purposes. For production use, replace with real network capture data (e.g. CIC-IDS2017 or NSL-KDD datasets).

---
*Kunal's Lab · AI & ML Systems · For research and educational purposes only.*
