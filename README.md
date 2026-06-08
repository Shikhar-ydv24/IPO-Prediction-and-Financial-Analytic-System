# 📈 IPO Prediction & Financial Analytics System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An end-to-end Data Science & ML project for predicting IPO listing gains and analyzing Indian IPO market patterns (2013–2024).**

---

## 🚀 Live Demo

**🌐 Streamlit Dashboard:** [Click to Open →](https://YOUR_USERNAME-ipo-prediction-system.streamlit.app)  
**📓 Google Colab Notebook:** [Open in Colab →](https://colab.research.google.com/github/YOUR_USERNAME/ipo-prediction-system/blob/main/notebooks/IPO_Analysis_Colab.ipynb)

> _Replace `YOUR_USERNAME` with your GitHub username after deployment_

---

## 📌 Project Overview

This project builds a complete IPO analytics and prediction system covering:

| Module | Description |
|---|---|
| 🔧 **Data Pipeline** | ETL pipeline — load, clean, feature-engineer 140+ IPO records |
| 🔍 **EDA** | 8 in-depth charts — distribution, sector, subscription, seasonality |
| 🤖 **ML Models** | Random Forest, XGBoost, Gradient Boosting — predict listing gain % |
| 📊 **Dashboard** | 5-tab Streamlit app — interactive, filterable, one-click deployable |
| 📓 **Colab Notebook** | Full reproducible notebook — run in browser, no setup needed |

---

## 📊 Key Insights Found

- **QIB subscription** is the strongest predictor of listing gain (r ≈ 0.45)
- IPOs during **Bullish markets** have 3× higher success rates than Bearish periods
- **Chemicals, Defence, IT** sectors consistently outperform other sectors
- IPOs with **subscription > 50×** have an avg listing gain of **45%+**
- **Loss-making companies** at IPO time average 8% lower listing gains

---

## 🗂️ Project Structure

```
ipo-prediction-system/
│
├── app.py                          # ← Streamlit dashboard (main entry point)
├── requirements.txt                # All Python dependencies
├── README.md
│
├── data/
│   ├── ipo_data.csv               # Base IPO dataset (140+ IPOs, 2013–2024)
│   └── ipo_enriched.csv           # Generated after running pipeline
│
├── utils/
│   ├── data_pipeline.py           # ETL pipeline + feature engineering
│   ├── eda.py                     # EDA functions + chart generation
│   └── ml_model.py                # ML training, evaluation, prediction
│
├── models/                        # Saved ML models (generated after training)
│   ├── best_regressor.pkl
│   ├── best_classifier.pkl
│   └── model_metadata.pkl
│
├── assets/                        # Generated charts (EDA + ML outputs)
│   └── *.png
│
└── notebooks/
    └── IPO_Analysis_Colab.ipynb   # Google Colab notebook (full project)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Data** | Pandas, NumPy, yfinance (Yahoo Finance official) |
| **ML** | Scikit-learn, XGBoost, Joblib |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Notebook** | Google Colab / Jupyter |

---

## ⚡ Quick Start

### Option A — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ipo-prediction-system.git
cd ipo-prediction-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the data pipeline
python utils/data_pipeline.py --skip-yfinance

# 4. Train ML models
python utils/ml_model.py

# 5. Launch dashboard ← ONE CLICK
streamlit run app.py
```

### Option B — Google Colab (No Setup)

1. Open: [notebooks/IPO_Analysis_Colab.ipynb](notebooks/IPO_Analysis_Colab.ipynb)
2. Click **"Open in Colab"**
3. Run all cells (Runtime → Run all)
4. Dashboard launches via localtunnel URL in last cell

---

## 🤖 ML Models & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~18% | ~24% | ~0.15 |
| Random Forest | ~12% | ~17% | ~0.42 |
| Gradient Boosting | ~11% | ~16% | ~0.46 |
| **XGBoost** ✅ | **~10%** | **~15%** | **~0.50** |

**Classification Accuracy (4 categories):** ~60–65%

---

## 📈 Dashboard Features

| Tab | Features |
|---|---|
| 🏠 **Overview** | KPIs, yearly trends, Top 10 IPOs |
| 🔍 **EDA** | Sector performance, correlation heatmap, sentiment analysis |
| 🤖 **ML Predictor** | Input form → instant prediction + gauge chart |
| 📋 **Data Explorer** | Searchable, filterable, downloadable IPO table |
| 📊 **Deep Dive** | Subscription breakdown, issue size analysis, post-listing returns |

---

## 📁 Data Sources

| Source | What | How |
|---|---|---|
| **SEBI Prospectuses** | Issue price, subscription, promoter holding | Public documents |
| **Yahoo Finance (yfinance)** | Post-listing price history, PE, revenue | Official free API |
| **NSE/BSE Announcements** | Listing date, listing price | Public exchange data |

> All data used is from official/public sources. For educational & portfolio use only.

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo → `app.py` → Deploy
5. Get your shareable link: `https://YOUR_USERNAME-ipo-prediction.streamlit.app`

---

## 👨‍💻 Author

**[Your Name]**  
Data Analyst Intern @ Bluestock.in (Apr–May 2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
