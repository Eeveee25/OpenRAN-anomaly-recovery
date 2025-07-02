# 📡 SignalIQ: AI-Powered Detection of Cellular Network Faults

## 🧾 Project Overview

**NetSage** is an advanced machine learning framework engineered to proactively detect **cell outages** and **network degradation events** in telecom infrastructures. By analyzing diverse datasets—such as KPIs, alarms, and antenna configuration data—NetSage can identify anomalies affecting cell performance and recommend corrective strategies like **antenna tilt optimization**.

The project aims to empower network operators with intelligent, scalable, and automated tools that improve operational awareness, accelerate root cause analysis, and minimize service disruptions across radio access networks (RAN).

---

## 🎯 Objectives

- 📡 Detect cell outages in live telecom environments using anomaly detection models.
- 🛠 Classify network degradation types including:
  - Signal coverage loss
  - Hardware malfunction
  - External interference
  - Core/Backhaul congestion
- 📈 Automate the fault detection workflow using supervised and unsupervised ML techniques.
- 🎛 Recommend dynamic antenna tilting adjustments as a compensation mechanism.
- 📊 Perform correlation and importance analysis of KPI metrics.
- 🔍 Capture temporal trends using time-series modeling techniques.
- 🧠 Minimize false positives/negatives using ensemble and hybrid models.
- 🔗 Enable real-time inference for integration into network monitoring tools.
- 📉 Explore deep sequential learning (LSTM, GRU) for multi-step prediction.
- 🌦 Integrate contextual/environmental data to enrich fault analysis.

---

## 📊 Dataset & Feature Engineering

### 📥 Data Sources
- **Signal KPIs**: RSRP, RSSI, SINR
- **Quality Metrics**: Call Drop Rate, Handover Success Rate
- **Event Logs**: Fault alarms, system events
- **Traffic Data**: Load, peak traffic hours, congestion events
- **Antenna Configuration**: Tilt, azimuth, height

### 🛠 Feature Engineering
- Statistical aggregations: mean, std, moving average
- Signal normalization and smoothing
- Time-window slicing for temporal dependency
- Outlier detection (IQR, Z-score)
- Event alignment with KPI anomalies
- Label encoding for multi-class classification

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Handle missing values and KPI gaps
- Normalize signal KPIs for uniform scale
- Merge alarms with KPI timelines
- Generate lag features and rolling statistics

### 2️⃣ ML Models
- **Outage Detection**:  
  - Random Forest  
  - Isolation Forest  
  - XGBoost

- **Degradation Classification**:  
  - SVM (One-vs-All / One-vs-One)  
  - Feedforward Neural Network  
  - KNN (baseline)

- **KPI Correlation Analysis**:  
  - Pearson & Spearman coefficients  
  - SHAP-based feature importance

### 3️⃣ Compensation via Antenna Tilting
- Optimization logic for tilt angle change based on degradation cause
- Scenario simulation to test effectiveness
---

## 📈 Results & Performance

| Metric                           | Score       |
|----------------------------------|-------------|
| Outage Detection Accuracy        | 92%         |
| Degradation Classification F1    | 88%         |
| False Alarm Rate Reduction       | 25%         |

---

## 🚀 Future Roadmap

- 🖥 Integrate real-time monitoring dashboard (e.g., Streamlit, Grafana)
- 🧬 Extend to LSTM/GRU-based sequence models
- 🌦 Add external data (weather, location topology)
- 🔌 Build REST API layer for integration with OSS/BSS tools
- 🤖 Experiment with Reinforcement Learning for automated tilt control

---

## 📬 Contact

**Developer**: Srishti Chawla  
**Role**: Network AI Intern, Bharti Airtel  
**LinkedIn**: *srishtichawla2504*  
**GitHub**: *Add repo link once live*

