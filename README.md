# 🛡️ FraudShield AI — Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189B50?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-ready, end-to-end machine learning web application for detecting fraudulent credit card transactions in real time.**

[Features](#-features) · [Demo](#-app-pages) · [Installation](#-installation) · [Usage](#-usage) · [Models](#-machine-learning-models) · [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

FraudShield AI is a full-stack machine learning application built with **Streamlit** that detects credit card fraud using three powerful classifiers — Logistic Regression, Random Forest, and XGBoost. It features an interactive analytics dashboard, real-time transaction prediction, downloadable reports, and a premium dark UI with glassmorphism effects.

The dataset used is the **Kaggle Credit Card Fraud Detection dataset** — 284,807 real anonymized transactions from European cardholders, of which only 492 (0.17%) are fraudulent.

---

## ✨ Features

### 🎨 UI & Design
- Dark modern theme with deep navy gradients and purple/pink accents
- Glassmorphism cards with hover effects and smooth transitions
- Animated glowing result cards (red for fraud, green for genuine)
- Animated KPI metric cards with color-coded top borders
- Gauge meter for fraud probability visualization
- Fully responsive layout for desktop and mobile
- Custom CSS with `Plus Jakarta Sans` + `JetBrains Mono` typography
- Professional fintech-style dashboard feel

### 📊 Analytics Dashboard
- Total transactions, fraud count, genuine count, fraud percentage
- Total fraud monetary loss and average fraud amount
- Fraud vs. Genuine interactive pie chart (Plotly)
- Transaction amount distribution histogram (overlaid, clipped)
- Hourly fraud activity timeline (spline area chart)
- Average amount comparison by class (bar chart)
- Fraud transactions by amount bucket (gradient bar chart)

### 🔬 EDA & Insights
- V-feature mean comparison (Fraud vs. Genuine side-by-side)
- V1–V6 distribution histograms (overlaid per feature)
- Full feature correlation heatmap (15 features, diverging colorscale)
- Top 15 highest-amount fraud transactions (table + chart)
- Full statistical summary (describe) for fraud and genuine subsets

### 🤖 Machine Learning
- **3 Classifiers:** Logistic Regression · Random Forest · XGBoost
- Balanced training via under-sampling (492 fraud + 492 legit)
- 80/20 train-test split with stratification
- Metrics: Accuracy, Precision, Recall, F1 Score
- Interactive confusion matrix heatmap
- ROC curve with AUC for all 3 models on a single chart
- Side-by-side model comparison bar chart
- Loading progress bar during training

### 🎯 Prediction Engine
- Manual input for all 30 features (Time, V1–V28, Amount)
- One-click sample loaders: "Load Fraud Sample" / "Load Genuine Sample"
- Instant fraud probability with animated gauge meter
- Dual probability bars (fraud + legit)
- Glowing animated result cards (pulsing red/green)
- Prediction history log (last 10 predictions)
- Downloadable prediction report as JSON

### 📋 Data Explorer
- Filterable interactive dataframe (by class and amount range)
- Adjustable row count slider
- One-click CSV export of filtered data
- Missing value analysis
- Data type overview per column

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Project hero, architecture breakdown, tech stack |
| 📊 **Dashboard** | KPI cards, charts, fraud distribution visuals |
| 🔬 **EDA & Insights** | Feature analysis, heatmap, statistical summaries |
| 🤖 **ML Models** | Train models, view metrics, confusion matrix, ROC |
| 🎯 **Prediction** | Input a transaction and get a real-time fraud verdict |
| 📋 **Data Explorer** | Browse, filter, and export the dataset |

---

## 📁 Project Structure

```
fraudshield-ai/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── creditcard.csv          # Dataset (download separately from Kaggle)
└── README.md               # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/fraudshield-ai.git
cd fraudshield-ai
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

Download `creditcard.csv` from Kaggle:

👉 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place it in the same directory as `app.py`.

---

## 🚀 Usage

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Tip:** If `creditcard.csv` is not in the same folder, use the **Upload** button in the sidebar to load it at runtime.

---

## 🤖 Machine Learning Models

All models are trained on a **balanced dataset** created by under-sampling the majority class (genuine) to match the minority class (fraud) at 492 samples each.

| Model | Accuracy | F1 Score | Notes |
|-------|----------|----------|-------|
| Logistic Regression | ~93.9% | ~93.9% | Fast, interpretable baseline |
| Random Forest | ~95.9% | ~95.9% | Ensemble, handles non-linearity |
| XGBoost | ~96.5% | ~96.5% | Gradient boosting, best overall |

### Training Pipeline

```
Raw Dataset (284,807 rows)
        │
        ▼
Class Balancing (under-sample legit → 492 + 492 = 984 rows)
        │
        ▼
Train/Test Split (80% / 20%, stratified)
        │
        ▼
Fit: LR · Random Forest · XGBoost
        │
        ▼
Evaluate: Accuracy · Precision · Recall · F1 · AUC · Confusion Matrix
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — ULB Machine Learning Group |
| Total Transactions | 284,807 |
| Fraud Transactions | 492 (0.17%) |
| Genuine Transactions | 284,315 (99.83%) |
| Features | 30 (Time, V1–V28, Amount) |
| Target | Class (0 = Genuine, 1 = Fraud) |
| Time Window | 2 days |

> **Note:** Features V1–V28 are the result of PCA transformation to protect cardholder confidentiality. Only `Time` and `Amount` are in their original form.

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Enhancements

- [ ] Add SMOTE oversampling as an alternative balancing strategy
- [ ] Add SHAP / feature importance explainability charts
- [ ] Connect to a live transaction database (PostgreSQL / SQLite)
- [ ] Add email/SMS fraud alert notifications
- [ ] Deploy to Streamlit Cloud or AWS EC2
- [ ] Add user authentication for multi-user access
- [ ] Add model versioning and auto-retraining pipeline
- [ ] Support batch CSV upload for bulk prediction

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- 💼 LinkedIn: [linkedin.com](www.linkedin.com/in/prathamesh-nehete-8b27b8249)
- 🐙 GitHub: [github.com](https://github.com/Pratham2474)
- 📧 Email: prathameshnehete73@gmail.com

---

## 🙏 Acknowledgements

- Dataset by the [Machine Learning Group — ULB](https://mlg.ulb.ac.be/)
- Published on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by Andrea Dal Pozzolo et al.
- Built with [Streamlit](https://streamlit.io), [Scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io), and [Plotly](https://plotly.com)

---

<div align="center">
<b>⭐ If you found this project helpful, please give it a star on GitHub!</b>
</div>
