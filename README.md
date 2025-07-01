# 📡 OpenRAN Gym-Based Anomaly Detection and Auto-Recovery System

## 🔍 Project Overview

This repository presents a modular AI-driven framework for **anomaly detection** and **self-healing recovery logic** within a 5G Radio Access Network (RAN), using synthetic and simulated KPI data generated through [OpenRAN Gym](https://github.com/open-5g/openrangym). The project replicates a real-world telecom automation pipeline, incorporating RAN telemetry ingestion, unsupervised anomaly detection, and rule-based automated mitigation strategies — aligned with Open RAN and Self-Organizing Networks (SON) principles.

> ⚙️ This initiative was undertaken as part of a professional internship at Bharti Airtel Networks Division, with the goal of solving real-world telecom performance bottlenecks using cutting-edge, open-source RAN simulation environments.

---

## 🎯 Objectives

- Ingest and preprocess simulated RAN KPI datasets
- Normalize and clean multivendor/multiformat KPI logs
- Detect anomalous behavior using AI/ML models
- Simulate corrective recommendations (SON-inspired)
- Visualize network health in real time with a lightweight dashboard

---

## 🧠 Key Concepts & Technologies

| Domain Area             | Component |
|-------------------------|-----------|
| **Telecom**             | RSRP, RSRQ, SINR, PRB Utilization, DL Throughput |
| **AI/ML**               | Isolation Forest, AutoEncoders, Time-Series Models |
| **Self-Healing Networks** | xApp logic, rule-based remediation, anomaly prioritization |
| **Open RAN Simulation** | [OpenRAN Gym](https://openrangym.com), synthetic xApps |
| **Visualization**       | Streamlit / Dash for anomaly & status dashboards |

---

## 🏗 Architecture Overview

       +------------------------+
       |  Synthetic KPI Source  |
       |  (OpenRAN Gym Logs)    |
       +-----------+------------+
                   |
                   v
       +-----------+------------+
       |  Preprocessing Engine  |
       |  (Cleaner + Normalizer)|
       +-----------+------------+
                   |
                   v
       +-----------+------------+
       |  Anomaly Detection ML  |
       |  (Isolation Forest / AE)|
       +-----------+------------+
                   |
                   v
       +-----------+------------+
       | Recovery Recommender   |
       | (SON-Inspired Logic)   |
       +-----------+------------+
                   |
                   v
       +-----------+------------+
       |   Dashboard / Output   |
       +------------------------+

---

## 🧪 Sample KPIs (Reference)

| KPI           | Description                              |
|---------------|------------------------------------------|
| `RSRP`        | Reference Signal Received Power          |
| `RSRQ`        | Reference Signal Received Quality        |
| `SINR`        | Signal-to-Interference-plus-Noise Ratio  |
| `DL Throughput`| Downlink user data rate                 |
| `PRB Utilization`| Physical Resource Block Utilization    |

---

## 🛠 Tools & Dependencies

- **Python 3.8+**
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `tensorflow` (optional for AE)
- `streamlit` or `dash`
- `openrangym` (external repo integration)

> All dependencies will be tracked in `requirements.txt`.

---
## 📌 Motivation

Modern 5G networks are increasingly software-defined, data-rich, and complexity-driven. However, most AI/ML systems in telecom remain siloed in R&D. This project bridges the gap between **simulation** and **real-world field logic**, enabling future-ready telecom engineers to build, test, and demonstrate intelligent, scalable, and autonomous network solutions.

---

## ✨ Outcomes

- 🚀 Full-stack anomaly detection and auto-recovery demo using open tools
- 🧠 Practical ML for telecom KPIs in a SON-like loop
- 📊 Real-time visualization of network performance and health
- 💼 Portfolio-ready for careers in **network automation**, **AI Ops**, or **Open RAN R&D**

---

## 🤝 Acknowledgements

- [OpenRAN Gym](https://github.com/open-5g/openrangym) for their simulator framework
- Bharti Airtel Networks Department for the internship context
- Various open-source contributors in the telecom AI/ML space

---

## 📬 Contact

**Developer**: Srishti Chawla  
**Role**: Network AI Intern, Bharti Airtel  
**LinkedIn**: *srishtichawla2504*  
**GitHub**: *Add repo link once live*

---
