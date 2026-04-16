# GNSS Spoofing Detection using Multi-Agent AI System

## 1. Problem Understanding

Global Navigation Satellite Systems (GNSS) are vulnerable to spoofing attacks due to their low-power open signals. Spoofing introduces manipulated signals that cause incorrect position, velocity, or timing.

This project focuses on detecting spoofing attacks using an **unsupervised, AI-driven approach**, given only unlabeled GNSS signal data.

Key challenge:

* No labeled training data
* Detection must rely on anomaly patterns and signal inconsistencies

---

## 2. Data Understanding

Each timestamp contains 8 satellite channels (ch0–ch7), each representing a unique PRN.

Thus, each timestamp forms a **multi-satellite snapshot**, enabling:

* per-satellite analysis
* cross-satellite consistency checks
* temporal modeling

---

## 3. Feature Engineering

We engineered features based on GNSS signal physics:

### Per-Satellite Features

* Doppler drift (Δ Doppler)
* Pseudorange change
* Carrier phase continuity
* Rolling statistics
* Jump detection

### Signal Processing Features

* Correlator symmetry: |EC - LC|
* Power ratio: PIP / PQP
* TOW–RX time gap
* EC/LC ratios

### Cross-Satellite Features

* CN0 mean and variance
* Doppler consistency
* Pseudorange spread
* TCD variance

These features capture both **physical constraints and spoofing signatures**.

---

## 4. Model Architecture

We use a **multi-agent system**:

### Agent 1: Statistical Model

* Isolation Forest for global anomaly detection

### Agent 2: Temporal Model

* LSTM Autoencoder to detect sequence anomalies

### Agent 3: Local Pattern Model

* CNN Autoencoder for short-term temporal patterns

### Agent 4: Rule-Based Detector

* Physics-based thresholds (e.g., correlator symmetry)

---

## 5. Training Methodology

Since no labels are available:

* Assumed majority data is normal
* Used anomaly detection techniques
* Applied temporal sequence modeling
* Combined multiple models for robustness

---

## 6. Fusion Strategy

Predictions from multiple agents are combined using a score-based fusion approach:

- Anomaly scores from different models are aggregated into a final fusion score
- A threshold is applied to obtain binary predictions
- Temporal smoothing is applied using a rolling window to enforce continuity:

    final_spoofed = rolling_max(window=5)

This ensures that spoofing is detected as continuous segments rather than isolated spikes, improving robustness.

## Confidence Estimation

Confidence scores are computed using a sigmoid transformation of the fusion score:

    confidence = 1 / (1 + exp(-k * (fusion_score - threshold)))

where:
- fusion_score represents the combined anomaly score
- threshold is the decision boundary
- k controls sharpness (set to 10)

This maps anomaly scores into a probability-like range [0,1], where higher values indicate stronger confidence in spoofing.

## 7. Evaluation Strategy

Without labels, evaluation was performed using:

* Temporal clustering of anomalies
* Distribution of anomaly scores
* Cross-satellite consistency checks
* Synthetic anomaly injection

Detected spoofing events appeared as **continuous time clusters**, validating model behavior.

---

## 8. Results

* Model successfully identifies spoofing clusters
* Minimal random false positives
* Robust across temporal segments

---

## 9. How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 10. Output

The model generates:

```
outputs/submission.csv
```

Format:

* time
* spoofed (0/1)
* confidence (0–1)

---

## 11. Key Contributions

* Physics-aware feature engineering
* Multi-agent anomaly detection system
* Temporal modeling without labels
* Robust and scalable pipeline
