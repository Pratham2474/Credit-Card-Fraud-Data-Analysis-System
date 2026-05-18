# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         CREDIT CARD FRAUD DETECTION SYSTEM — Production Streamlit App       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
from xgboost import XGBClassifier
import warnings, io, base64, time, json
from datetime import datetime
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI · Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f1729 0%, #080c18 55%, #06090f 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1f 0%, #080c18 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c18; }
::-webkit-scrollbar-thumb { background: #2d3167; border-radius: 4px; }

/* ═══════════════════════════════════
   SIDEBAR BRAND
═══════════════════════════════════ */
.brand-block {
    background: linear-gradient(135deg, #1a1f3d, #111428);
    border-bottom: 1px solid rgba(99,102,241,0.2);
    padding: 22px 20px 18px;
    margin-bottom: 10px;
}
.brand-name {
    font-size: 1.35rem; font-weight: 800; color: #f0f1ff;
    letter-spacing: -0.3px; line-height: 1.2;
}
.brand-name span {
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-tagline { font-size: 0.72rem; color: #3d4270; margin-top: 4px; letter-spacing: 0.8px; text-transform: uppercase; }
.brand-badge {
    display: inline-block; margin-top: 10px;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.68rem; color: #818cf8; letter-spacing: 0.5px;
}

/* Nav items */
[data-testid="stSidebar"] .stRadio label {
    color: #5a6080 !important; font-size: 0.88rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label {
    color: #a5b4fc !important;
}

/* ═══════════════════════════════════
   HERO SECTION
═══════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, #0f1535 0%, #141b3d 40%, #0c1028 100%);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 20px;
    padding: 44px 52px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute;
    top: -100px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(129,140,248,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: ''; position: absolute;
    bottom: -60px; left: 20%;
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(192,132,252,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: #6366f1; margin-bottom: 10px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; color: #f0f1ff;
    letter-spacing: -1px; line-height: 1.1; margin: 0 0 10px 0;
}
.hero-title span {
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-desc { color: #4b5280; font-size: 1rem; max-width: 560px; line-height: 1.6; }
.hero-chips { display: flex; gap: 8px; margin-top: 20px; flex-wrap: wrap; }
.chip {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.75rem; color: #818cf8; font-weight: 500;
}

/* ═══════════════════════════════════
   KPI CARDS
═══════════════════════════════════ */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 24px; }
.kpi-card {
    background: linear-gradient(135deg, #0d1030 0%, #0a0d22 100%);
    border-radius: 16px; padding: 22px 24px;
    border: 1px solid rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
    transition: transform 0.25s ease, border-color 0.25s ease;
    cursor: default;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(99,102,241,0.35); }
.kpi-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.c1::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.c2::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.c3::before { background: linear-gradient(90deg, #ef4444, #f97316); }
.kpi-card.c4::before { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 10px; }
.kpi-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #3d4270; margin-bottom: 4px; }
.kpi-val { font-size: 2rem; font-weight: 800; color: #e8eaff; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.kpi-sub { font-size: 0.75rem; color: #3d4270; margin-top: 5px; }

/* ═══════════════════════════════════
   PANELS / CARDS
═══════════════════════════════════ */
.panel {
    background: linear-gradient(135deg, #0d1030 0%, #0a0d22 100%);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 24px 26px; margin-bottom: 20px;
}
.panel-title {
    font-size: 0.95rem; font-weight: 700; color: #c8cbe8;
    margin: 0 0 18px 0; display: flex; align-items: center; gap: 8px;
}
.panel-title .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #c084fc);
    display: inline-block;
    box-shadow: 0 0 8px rgba(99,102,241,0.6);
}

/* ═══════════════════════════════════
   SECTION DIVIDER
═══════════════════════════════════ */
.section-header {
    display: flex; align-items: center; gap: 12px; margin: 36px 0 20px;
}
.section-header .line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent); }
.section-header .text {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #6366f1; white-space: nowrap;
}

/* ═══════════════════════════════════
   METRIC BADGES
═══════════════════════════════════ */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.metric-badge {
    background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px; padding: 16px 18px; text-align: center;
    transition: all 0.2s;
}
.metric-badge:hover { background: rgba(99,102,241,0.12); border-color: rgba(99,102,241,0.3); }
.mb-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: #4b5280; margin-bottom: 6px; }
.mb-val { font-size: 1.7rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.mb-val.good { color: #34d399; }
.mb-val.warn { color: #fbbf24; }
.mb-val.info { color: #818cf8; }
.mb-val.accent { color: #c084fc; }

/* ═══════════════════════════════════
   PREDICTION RESULT CARDS
═══════════════════════════════════ */
.pred-fraud {
    background: linear-gradient(135deg, #1c0808, #200a0a);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px; padding: 36px 40px; text-align: center;
    box-shadow: 0 0 40px rgba(239,68,68,0.12), inset 0 0 40px rgba(239,68,68,0.03);
    animation: pulseRed 2s infinite;
}
.pred-legit {
    background: linear-gradient(135deg, #081c0e, #0a2012);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px; padding: 36px 40px; text-align: center;
    box-shadow: 0 0 40px rgba(16,185,129,0.12), inset 0 0 40px rgba(16,185,129,0.03);
    animation: pulseGreen 2s infinite;
}
@keyframes pulseRed {
    0%,100% { box-shadow: 0 0 30px rgba(239,68,68,0.1); }
    50% { box-shadow: 0 0 55px rgba(239,68,68,0.22); }
}
@keyframes pulseGreen {
    0%,100% { box-shadow: 0 0 30px rgba(16,185,129,0.1); }
    50% { box-shadow: 0 0 55px rgba(16,185,129,0.22); }
}
.pred-icon { font-size: 3.5rem; margin-bottom: 10px; line-height: 1; }
.pred-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.pred-fraud  .pred-label { color: #ef4444; }
.pred-legit  .pred-label { color: #10b981; }
.pred-title { font-size: 1.9rem; font-weight: 800; margin: 0 0 8px; }
.pred-fraud  .pred-title { color: #fca5a5; }
.pred-legit  .pred-title { color: #6ee7b7; }
.pred-body { color: #4b5280; font-size: 0.9rem; line-height: 1.6; }
.prob-bar-wrap { margin-top: 18px; }
.prob-bar-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: #4b5280; margin-bottom: 6px; }
.prob-bar-bg { background: rgba(255,255,255,0.05); border-radius: 8px; height: 10px; overflow: hidden; }
.prob-bar-fill-red  { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #ef4444, #f97316); transition: width 1s ease; }
.prob-bar-fill-green{ height: 100%; border-radius: 8px; background: linear-gradient(90deg, #10b981, #34d399); transition: width 1s ease; }

/* ═══════════════════════════════════
   BUTTONS
═══════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 12px 28px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ═══════════════════════════════════
   MISC STREAMLIT OVERRIDES
═══════════════════════════════════ */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1030, #0a0d22) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important; padding: 14px 18px !important;
}
div[data-testid="stMetric"] label { color: #3d4270 !important; font-size: 0.75rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e8eaff !important; font-family: 'JetBrains Mono', monospace !important;
}
.stSelectbox > div > div {
    background: #0d1030 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; color: #c8cbe8 !important;
}
.stSlider [data-baseweb="slider"] > div > div > div { background: #6366f1 !important; }
.stNumberInput input, .stTextInput input {
    background: #0d1030 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    color: #c8cbe8 !important; border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
h1,h2,h3,h4 { color: #c8cbe8 !important; }
p, li { color: #4b5280; }
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #3d4270 !important; font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 10px 22px !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1 !important; background: transparent !important;
}
.stExpander { border: 1px solid rgba(99,102,241,0.15) !important; border-radius: 12px !important; background: #0d1030 !important; }
.stExpander summary { color: #818cf8 !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ═══════════════════════════════════
   FOOTER
═══════════════════════════════════ */
.footer {
    margin-top: 60px;
    background: linear-gradient(135deg, #0a0d22, #070910);
    border: 1px solid rgba(99,102,241,0.12); border-radius: 16px;
    padding: 32px 40px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 16px;
}
.footer-left { }
.footer-brand { font-size: 1.1rem; font-weight: 800; color: #e8eaff; }
.footer-brand span { background: linear-gradient(90deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.footer-copy { font-size: 0.75rem; color: #2d3060; margin-top: 4px; }
.footer-links { display: flex; gap: 12px; flex-wrap: wrap; }
.footer-link {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px; padding: 7px 16px;
    font-size: 0.78rem; color: #818cf8; font-weight: 600;
    text-decoration: none; transition: all 0.2s;
}
.footer-link:hover { background: rgba(99,102,241,0.18); color: #c084fc; }
.footer-stack { font-size: 0.72rem; color: #2d3060; }

/* ═══════════════════════════════════
   GLASSMORPHISM CARD
═══════════════════════════════════ */
.glass {
    background: rgba(13,16,48,0.6);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 16px; padding: 24px;
}

/* Insight row */
.insight-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 20px; }
.insight-card {
    background: rgba(13,16,48,0.8); border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px; padding: 20px; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.insight-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.3); }
.insight-card .ic-icon { font-size: 1.8rem; margin-bottom: 8px; }
.insight-card .ic-val { font-size: 1.5rem; font-weight: 800; color: #e8eaff; font-family: 'JetBrains Mono',monospace; }
.insight-card .ic-label { font-size: 0.72rem; color: #3d4270; margin-top: 3px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

/* Sidebar nav */
.nav-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2d3060; padding: 12px 16px 4px; }
.sidebar-divider { height: 1px; background: rgba(99,102,241,0.1); margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color='#4b5280', size=12),
    margin=dict(t=36, b=36, l=16, r=16),
    xaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.12)',
               tickfont=dict(color='#3d4270'), zerolinecolor='rgba(99,102,241,0.08)'),
    yaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.12)',
               tickfont=dict(color='#3d4270'), zerolinecolor='rgba(99,102,241,0.08)'),
    legend=dict(font=dict(color='#5a6080'), bgcolor='rgba(0,0,0,0)'),
)
C = dict(fraud='#ef4444', legit='#6366f1', accent='#c084fc',
         green='#10b981', amber='#f59e0b', pink='#f472b6',
         blue='#60a5fa')
GRAD_FRAUD  = [[0,'#6366f1'],[0.5,'#ec4899'],[1,'#ef4444']]
GRAD_LEGIT  = [[0,'#6366f1'],[1,'#8b5cf6']]

# ──────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & CACHING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    return df

@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    """Train all three models on a balanced dataset and return metrics."""
    df = st.session_state['df']
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    balanced = pd.concat([legit.sample(n=len(fraud), random_state=42), fraud])
    X = balanced.drop(columns='Class')
    y = balanced['Class']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = {}
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':             XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0),
    }
    for name, clf in classifiers.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(yte, prob)
        results[name] = dict(
            model=clf, pred=pred, prob=prob,
            acc=accuracy_score(yte, pred),
            prec=precision_score(yte, pred, zero_division=0),
            rec=recall_score(yte, pred, zero_division=0),
            f1=f1_score(yte, pred, zero_division=0),
            cm=confusion_matrix(yte, pred),
            fpr=fpr, tpr=tpr, auc=auc(fpr, tpr),
            Xte=Xte, yte=yte,
        )
    return results

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
if 'df' not in st.session_state: st.session_state['df'] = None
if 'models_trained' not in st.session_state: st.session_state['models_trained'] = False
if 'results' not in st.session_state: st.session_state['results'] = None
if 'pred_log' not in st.session_state: st.session_state['pred_log'] = []

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
        <div class="brand-name">🛡️ <span>FraudShield</span> AI</div>
        <div class="brand-tagline">Credit Card Intelligence Platform</div>
        <div class="brand-badge">v2.0 · Production Ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📊  Dashboard",
        "🔬  EDA & Insights",
        "🤖  ML Models",
        "🎯  Prediction",
        "📋  Data Explorer",
    ], label_visibility="collapsed")
    page = page.split("  ")[1].strip()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload creditcard.csv", type=['csv'], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Loading dataset..."):
            st.session_state['df'] = load_data(uploaded)
            st.session_state['models_trained'] = False
        st.success(f"✅ {len(st.session_state['df']):,} rows loaded")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:8px 0 4px;font-size:0.68rem;color:#2d3060;text-align:center">Built with Streamlit · Scikit-learn · XGBoost<br>© 2025 FraudShield AI</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  AUTO-LOAD LOCAL FILE IF AVAILABLE
# ──────────────────────────────────────────────────────────────────────────────
import os
if st.session_state['df'] is None:
    for candidate in ['creditcard.csv', '/mnt/user-data/uploads/creditcard.csv']:
        if os.path.exists(candidate):
            st.session_state['df'] = load_data(candidate)
            break

# ──────────────────────────────────────────────────────────────────────────────
#  DERIVED STATS (used across pages)
# ──────────────────────────────────────────────────────────────────────────────
df = st.session_state['df']
if df is not None:
    total     = len(df)
    n_fraud   = int(df['Class'].sum())
    n_legit   = total - n_fraud
    fraud_pct = n_fraud / total * 100
    avg_amt_fraud = df[df.Class==1]['Amount'].mean()
    avg_amt_legit = df[df.Class==0]['Amount'].mean()
    total_loss    = df[df.Class==1]['Amount'].sum()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">🛡️ AI-Powered Security Intelligence</div>
        <h1 class="hero-title">Credit Card<br><span>Fraud Detection</span><br>System</h1>
        <p class="hero-desc">
            An end-to-end machine learning platform that detects fraudulent credit card transactions
            in real time using Logistic Regression, Random Forest, and XGBoost — with interactive
            analytics and instant predictions.
        </p>
        <div class="hero-chips">
            <span class="chip">📊 284,807 Transactions</span>
            <span class="chip">🧬 3 ML Models</span>
            <span class="chip">⚡ Real-time Prediction</span>
            <span class="chip">🎯 ~95%+ Accuracy</span>
            <span class="chip">📈 ROC / AUC Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span class="dot"></span>Project Architecture</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                <div class="glass" style="padding:16px">
                    <div style="font-size:1.4rem;margin-bottom:6px">📥</div>
                    <div style="font-size:0.82rem;font-weight:700;color:#c8cbe8">Data Ingestion</div>
                    <div style="font-size:0.72rem;color:#3d4270;margin-top:4px">284K real transactions. PCA-anonymized V1–V28 features + Amount + Time</div>
                </div>
                <div class="glass" style="padding:16px">
                    <div style="font-size:1.4rem;margin-bottom:6px">⚖️</div>
                    <div style="font-size:0.82rem;font-weight:700;color:#c8cbe8">Class Balancing</div>
                    <div style="font-size:0.72rem;color:#3d4270;margin-top:4px">Under-sampling legit transactions to match 492 fraud cases for fair training</div>
                </div>
                <div class="glass" style="padding:16px">
                    <div style="font-size:1.4rem;margin-bottom:6px">🤖</div>
                    <div style="font-size:0.82rem;font-weight:700;color:#c8cbe8">Model Training</div>
                    <div style="font-size:0.72rem;color:#3d4270;margin-top:4px">LR · Random Forest · XGBoost — trained & evaluated with full metrics</div>
                </div>
                <div class="glass" style="padding:16px">
                    <div style="font-size:1.4rem;margin-bottom:6px">🎯</div>
                    <div style="font-size:0.82rem;font-weight:700;color:#c8cbe8">Prediction Engine</div>
                    <div style="font-size:0.72rem;color:#3d4270;margin-top:4px">Input any transaction → get instant fraud probability + confidence score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title"><span class="dot"></span>Technology Stack</div>
        """, unsafe_allow_html=True)
        stack = [
            ("🐍", "Python 3.10+",      "Core runtime"),
            ("🎈", "Streamlit",          "Web framework"),
            ("🐼", "Pandas / NumPy",     "Data processing"),
            ("🔬", "Scikit-learn",       "ML algorithms"),
            ("⚡", "XGBoost",            "Gradient boosting"),
            ("📊", "Plotly",             "Interactive charts"),
            ("🎨", "Custom CSS",         "UI/UX styling"),
        ]
        for icon, name, desc in stack:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid rgba(99,102,241,0.08)">
                <span style="font-size:1.1rem;width:24px">{icon}</span>
                <div>
                    <div style="font-size:0.82rem;font-weight:700;color:#c8cbe8">{name}</div>
                    <div style="font-size:0.7rem;color:#3d4270">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset needed notice
    if df is None:
        st.info("⬅️  Please upload **creditcard.csv** from the sidebar to unlock all features.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    if df is None:
        st.warning("Please upload creditcard.csv to continue.")
        st.stop()

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c1">
            <div class="kpi-icon">💳</div>
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-val">{total:,}</div>
            <div class="kpi-sub">2-day capture window</div>
        </div>
        <div class="kpi-card c2">
            <div class="kpi-icon">✅</div>
            <div class="kpi-label">Genuine</div>
            <div class="kpi-val">{n_legit:,}</div>
            <div class="kpi-sub">{100-fraud_pct:.2f}% of all transactions</div>
        </div>
        <div class="kpi-card c3">
            <div class="kpi-icon">🚨</div>
            <div class="kpi-label">Fraudulent</div>
            <div class="kpi-val">{n_fraud}</div>
            <div class="kpi-sub">{fraud_pct:.4f}% fraud rate</div>
        </div>
        <div class="kpi-card c4">
            <div class="kpi-icon">💸</div>
            <div class="kpi-label">Total Fraud Loss</div>
            <div class="kpi-val">${total_loss:,.0f}</div>
            <div class="kpi-sub">Avg ${avg_amt_fraud:.2f} per fraud</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Pie + Amount histogram ────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Transaction Class Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Genuine', 'Fraudulent'],
            values=[n_legit, n_fraud],
            hole=0.68,
            marker=dict(colors=[C['legit'], C['fraud']],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            textinfo='label+percent',
            textfont=dict(color='#c8cbe8', size=12),
            pull=[0, 0.04],
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
        ))
        fig.add_annotation(
            text=f'<b style="font-size:18px">{total:,}</b><br><span style="font-size:11px">Transactions</span>',
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='#c8cbe8', size=14, family='JetBrains Mono')
        )
        fig.update_layout(**BASE_LAYOUT, height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df[df.Class==0]['Amount'].clip(upper=1000),
            name='Genuine', marker_color=C['legit'], opacity=0.65, nbinsx=60,
            hovertemplate='Amount: $%{x}<br>Count: %{y:,}<extra></extra>'
        ))
        fig2.add_trace(go.Histogram(
            x=df[df.Class==1]['Amount'].clip(upper=1000),
            name='Fraud', marker_color=C['fraud'], opacity=0.85, nbinsx=60,
            hovertemplate='Amount: $%{x}<br>Count: %{y:,}<extra></extra>'
        ))
        fig2.update_layout(**BASE_LAYOUT, height=300, barmode='overlay',
                           xaxis_title='Amount ($)', yaxis_title='Count')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Hourly + Amount comparison ────────────────────────────────────
    col3, col4 = st.columns([1.5, 1])

    with col3:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Fraud Activity Over Time (Hourly)</div>', unsafe_allow_html=True)
        fdf = df[df.Class==1].copy()
        fdf['hour'] = (fdf['Time'] // 3600).astype(int)
        hourly = fdf.groupby('hour').size().reset_index(name='count')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['count'],
            mode='lines+markers',
            line=dict(color=C['accent'], width=2.5, shape='spline'),
            marker=dict(color=C['accent'], size=5,
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            fill='tozeroy', fillcolor='rgba(192,132,252,0.06)',
            hovertemplate='Hour %{x}<br>Fraud cases: %{y}<extra></extra>'
        ))
        fig3.update_layout(**BASE_LAYOUT, height=280,
                           xaxis_title='Hour', yaxis_title='Fraud Count')
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Avg Amount by Class</div>', unsafe_allow_html=True)
        fig4 = go.Figure(go.Bar(
            x=['Genuine', 'Fraudulent'],
            y=[avg_amt_legit, avg_amt_fraud],
            marker=dict(
                color=[avg_amt_legit, avg_amt_fraud],
                colorscale=[[0,C['legit']],[1,C['fraud']]],
                showscale=False,
                line=dict(width=0)
            ),
            text=[f'${avg_amt_legit:.2f}', f'${avg_amt_fraud:.2f}'],
            textposition='outside', textfont=dict(color='#c8cbe8', size=13,
                                                   family='JetBrains Mono'),
            width=0.45,
            hovertemplate='%{x}<br>Avg Amount: $%{y:.2f}<extra></extra>'
        ))
        fig4.update_layout(**BASE_LAYOUT, height=280,
                           yaxis_title='Average Amount ($)')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Fraud by amount bucket ────────────────────────────────────────────────
    st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Fraud Transactions by Amount Range</div>', unsafe_allow_html=True)
    bins   = [0, 25, 50, 100, 200, 500, 1000, 5000]
    labels = ['$0-25','$26-50','$51-100','$101-200','$201-500','$501-1K','$1K+']
    frd = df[df.Class==1].copy()
    frd['bucket'] = pd.cut(frd['Amount'], bins=bins, labels=labels, include_lowest=True)
    bc = frd['bucket'].value_counts().sort_index()
    fig5 = go.Figure(go.Bar(
        x=bc.index.tolist(), y=bc.values,
        marker=dict(
            color=bc.values,
            colorscale=[[0,C['legit']],[0.5,C['accent']],[1,C['fraud']]],
            showscale=False, line=dict(width=0)
        ),
        text=bc.values, textposition='outside',
        textfont=dict(color='#c8cbe8', size=12),
        hovertemplate='%{x}<br>Fraud cases: %{y}<extra></extra>'
    ))
    fig5.update_layout(**BASE_LAYOUT, height=260,
                       xaxis_title='Amount Range', yaxis_title='Fraud Count')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA & Insights":
    if df is None: st.warning("Upload dataset first."); st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Feature Analysis", "Correlation Heatmap", "High-Risk Transactions", "Statistical Summary"])

    with tab1:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>V-Feature Mean Comparison: Fraud vs Genuine</div>', unsafe_allow_html=True)
        v_cols = [f'V{i}' for i in range(1, 15)]
        fraud_means  = df[df.Class==1][v_cols].mean()
        legit_means  = df[df.Class==0][v_cols].mean()
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(name='Fraudulent', x=v_cols, y=fraud_means,
                             marker_color=C['fraud'], opacity=0.85))
        fig.add_trace(go.Bar(name='Genuine', x=v_cols, y=legit_means,
                             marker_color=C['legit'], opacity=0.75))
        fig.update_layout(**BASE_LAYOUT, barmode='group', height=320,
                          yaxis_title='Mean Value')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>V-Feature Distributions (V1–V6) — Fraud vs Genuine</div>', unsafe_allow_html=True)
        cols_show = ['V1','V2','V3','V4','V5','V6']
        fig2 = make_subplots(rows=2, cols=3, subplot_titles=cols_show)
        for i, col in enumerate(cols_show):
            r, c_ = divmod(i, 3)
            fig2.add_trace(go.Histogram(x=df[df.Class==0][col], name='Genuine',
                                        marker_color=C['legit'], opacity=0.6, showlegend=(i==0)), row=r+1, col=c_+1)
            fig2.add_trace(go.Histogram(x=df[df.Class==1][col], name='Fraud',
                                        marker_color=C['fraud'], opacity=0.8, showlegend=(i==0)), row=r+1, col=c_+1)
        fig2.update_layout(**BASE_LAYOUT, height=400, barmode='overlay')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Feature Correlation Heatmap (Top 15 Features)</div>', unsafe_allow_html=True)
        top_cols = ['V1','V2','V3','V4','V7','V10','V11','V12','V14','V16','V17','V18','V19','Amount','Class']
        corr = df[top_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[
                [0.0, '#ef4444'], [0.25, '#7f1d1d'],
                [0.5, '#0d1030'],
                [0.75, '#1e3a8a'], [1.0, '#6366f1']
            ],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=8, color='rgba(255,255,255,0.5)'),
            hovertemplate='%{y} × %{x}<br>Corr: %{z:.3f}<extra></extra>',
            showscale=True,
            colorbar=dict(tickfont=dict(color='#3d4270'), thickness=12)
        ))
        fig_h.update_layout(**BASE_LAYOUT, height=480)
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Top 15 Highest-Amount Fraud Transactions</div>', unsafe_allow_html=True)
        top15 = df[df.Class==1].nlargest(15, 'Amount')[['Time','Amount','V1','V2','V3','V4']].reset_index(drop=True)
        top15.index += 1
        top15['Time_hr'] = top15['Time'].apply(lambda x: f"{int(x//3600):02d}h {int((x%3600)//60):02d}m")
        top15['Amount_fmt'] = top15['Amount'].apply(lambda x: f"${x:,.2f}")
        fig_t15 = go.Figure(go.Bar(
            x=[f"#{i+1}" for i in range(15)],
            y=top15['Amount'].values,
            marker=dict(
                color=top15['Amount'].values,
                colorscale=[[0,'#6366f1'],[1,'#ef4444']],
                showscale=False, line=dict(width=0)
            ),
            text=top15['Amount_fmt'].values, textposition='outside',
            textfont=dict(color='#c8cbe8', size=10),
            hovertemplate='Transaction #%{x}<br>Amount: $%{y:,.2f}<extra></extra>'
        ))
        fig_t15.update_layout(**BASE_LAYOUT, height=300, yaxis_title='Amount ($)')
        st.plotly_chart(fig_t15, use_container_width=True)

        disp = top15[['Time_hr','Amount_fmt','V1','V2','V3','V4']].rename(
            columns={'Time_hr':'Time','Amount_fmt':'Amount'})
        st.dataframe(disp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Dataset Statistical Summary</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Fraud Transactions — Amount Stats**")
            st.dataframe(df[df.Class==1]['Amount'].describe().to_frame().T.round(4),
                         use_container_width=True)
        with col_b:
            st.markdown("**Genuine Transactions — Amount Stats**")
            st.dataframe(df[df.Class==0]['Amount'].describe().to_frame().T.round(4),
                         use_container_width=True)

        st.markdown("**All Features Summary**")
        st.dataframe(df.describe().round(4), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Models":
    if df is None: st.warning("Upload dataset first."); st.stop()

    st.markdown("""
    <div class="panel">
        <div class="panel-title"><span class="dot"></span>Model Training</div>
        <p style="color:#3d4270;font-size:0.88rem;margin-bottom:0">
        Three classifiers are trained on a balanced dataset (under-sampled to 492 legit + 492 fraud).
        80/20 train-test split with stratification. Select a model and view its detailed metrics below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        model_choice = st.selectbox(
            "Select Model",
            ['Logistic Regression', 'Random Forest', 'XGBoost'],
            key='model_select'
        )
        train_btn = st.button("🚀  Train All Models", use_container_width=True)

    if train_btn or st.session_state['models_trained']:
        if train_btn:
            prog = st.progress(0)
            status = st.empty()
            for i, msg in enumerate([
                "Balancing dataset...", "Training Logistic Regression...",
                "Training Random Forest...", "Training XGBoost...", "Computing metrics..."
            ]):
                status.markdown(f"<p style='color:#818cf8;font-size:0.85rem'>⚙️ {msg}</p>", unsafe_allow_html=True)
                prog.progress((i+1)*20)
                time.sleep(0.3)
            results = train_models(id(df))
            st.session_state['results'] = results
            st.session_state['models_trained'] = True
            prog.empty(); status.empty()
            st.success("✅ All models trained successfully!")
        else:
            if st.session_state['results'] is None:
                results = train_models(id(df))
                st.session_state['results'] = results
            results = st.session_state['results']

        res = results[model_choice]

        # ── Metric badges ──────────────────────────────────────────────────
        st.markdown('<div class="section-header"><div class="line"></div><div class="text">Performance Metrics</div><div class="line"></div></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-badge">
                <div class="mb-label">Accuracy</div>
                <div class="mb-val good">{res['acc']*100:.2f}%</div>
            </div>
            <div class="metric-badge">
                <div class="mb-label">Precision</div>
                <div class="mb-val info">{res['prec']*100:.2f}%</div>
            </div>
            <div class="metric-badge">
                <div class="mb-label">Recall</div>
                <div class="mb-val warn">{res['rec']*100:.2f}%</div>
            </div>
            <div class="metric-badge">
                <div class="mb-label">F1 Score</div>
                <div class="mb-val accent">{res['f1']*100:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confusion matrix + ROC ─────────────────────────────────────────
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Confusion Matrix</div>', unsafe_allow_html=True)
            cm = res['cm']
            labels_cm = ['Genuine (0)', 'Fraud (1)']
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Predicted Genuine', 'Predicted Fraud'],
                y=['Actual Genuine', 'Actual Fraud'],
                colorscale=[[0,'#0a0d22'],[0.5,'#312e81'],[1,'#6366f1']],
                showscale=False,
                text=cm, texttemplate='<b>%{text}</b>',
                textfont=dict(size=22, color='#e8eaff', family='JetBrains Mono'),
                hovertemplate='%{y} → %{x}<br>Count: %{z}<extra></extra>'
            ))
            fig_cm.update_layout(**BASE_LAYOUT, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_roc:
            st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>ROC Curve</div>', unsafe_allow_html=True)
            # Draw ROC for all models on same chart
            fig_roc = go.Figure()
            colors_roc = [C['legit'], C['accent'], C['fraud']]
            for i, (mname, mres) in enumerate(results.items()):
                fig_roc.add_trace(go.Scatter(
                    x=mres['fpr'], y=mres['tpr'],
                    mode='lines',
                    name=f"{mname} (AUC={mres['auc']:.3f})",
                    line=dict(color=colors_roc[i], width=2.5,
                              dash='solid' if mname==model_choice else 'dot'),
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode='lines',
                name='Random', line=dict(color='#1e2040', width=1.5, dash='dash')
            ))
            fig_roc.update_layout(**BASE_LAYOUT, height=300,
                                   xaxis_title='False Positive Rate',
                                   yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Model comparison bar ───────────────────────────────────────────
        st.markdown('<div class="section-header"><div class="line"></div><div class="text">Model Comparison</div><div class="line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>All Models — Side-by-Side Metrics</div>', unsafe_allow_html=True)
        model_names = list(results.keys())
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy':  [results[m]['acc']  for m in model_names],
            'Precision': [results[m]['prec'] for m in model_names],
            'Recall':    [results[m]['rec']  for m in model_names],
            'F1 Score':  [results[m]['f1']   for m in model_names],
            'AUC':       [results[m]['auc']  for m in model_names],
        })
        fig_cmp = go.Figure()
        for col_m, color_m in [('Accuracy', C['legit']),('Precision', C['blue']),
                                ('Recall', C['amber']),('F1 Score', C['accent']),('AUC', C['pink'])]:
            fig_cmp.add_trace(go.Bar(
                name=col_m, x=model_names, y=metrics_df[col_m],
                marker_color=color_m, opacity=0.85,
                text=[f'{v:.3f}' for v in metrics_df[col_m]],
                textposition='outside', textfont=dict(size=9, color='#c8cbe8'),
                hovertemplate=f'{col_m}: %{{y:.4f}}<extra></extra>'
            ))
        fig_cmp.update_layout(**BASE_LAYOUT, barmode='group', height=320,
                              yaxis_range=[0, 1.12], yaxis_title='Score')
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Store best model for prediction
        best = max(results, key=lambda m: results[m]['f1'])
        st.session_state['best_model_name'] = best
        st.session_state['best_model']      = results[best]['model']
        for mname, mres in results.items():
            st.session_state[f'model_{mname}'] = mres['model']

    else:
        st.info("👆  Click **Train All Models** to start training.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    if df is None: st.warning("Upload dataset first."); st.stop()

    st.markdown("""
    <div class="panel">
        <div class="panel-title"><span class="dot"></span>Real-Time Fraud Prediction</div>
        <p style="color:#3d4270;font-size:0.88rem">
        Enter a transaction's features manually or load a sample.
        The trained model will score the transaction and return a fraud probability instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ensure models are trained
    if not st.session_state['models_trained'] or st.session_state['results'] is None:
        with st.spinner("Training models…"):
            results = train_models(id(df))
            st.session_state['results'] = results
            st.session_state['models_trained'] = True
            for mname, mres in results.items():
                st.session_state[f'model_{mname}'] = mres['model']

    results = st.session_state['results']

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1,1,1])
    with ctrl1:
        pred_model = st.selectbox("Model", list(results.keys()), key='pred_model_sel')
    with ctrl2:
        load_sample = st.button("📥  Load Fraud Sample")
    with ctrl3:
        load_legit = st.button("📥  Load Genuine Sample")

    # Sample values
    sample_fraud_vals = dict(zip(
        [f'V{i}' for i in range(1,29)] + ['Amount','Time'],
        df[df.Class==1].sample(1, random_state=99).drop(columns='Class').values.flatten().tolist()
    ))
    sample_legit_vals = dict(zip(
        [f'V{i}' for i in range(1,29)] + ['Amount','Time'],
        df[df.Class==0].sample(1, random_state=7).drop(columns='Class').values.flatten().tolist()
    ))
    if load_sample:
        st.session_state['sample'] = sample_fraud_vals
    if load_legit:
        st.session_state['sample'] = sample_legit_vals

    sv = st.session_state.get('sample', {})

    # ── Input grid ────────────────────────────────────────────────────────────
    with st.expander("⚙️  Transaction Feature Inputs", expanded=True):
        c1, c2, c3 = st.columns(3)
        input_vals = {}
        v_cols = [f'V{i}' for i in range(1,29)]
        with c1:
            input_vals['Time']   = st.number_input("Time (s)", value=float(sv.get('Time', 40000.0)), format="%.2f")
            input_vals['Amount'] = st.number_input("Amount ($)", value=float(sv.get('Amount', 50.0)), min_value=0.0, format="%.2f")
            for col in v_cols[:10]:
                input_vals[col] = st.number_input(col, value=float(sv.get(col, 0.0)), format="%.4f", key=f'inp_{col}')
        with c2:
            for col in v_cols[10:20]:
                input_vals[col] = st.number_input(col, value=float(sv.get(col, 0.0)), format="%.4f", key=f'inp_{col}')
        with c3:
            for col in v_cols[20:]:
                input_vals[col] = st.number_input(col, value=float(sv.get(col, 0.0)), format="%.4f", key=f'inp_{col}')

    predict_btn = st.button("🔍  Analyze Transaction", use_container_width=True)

    if predict_btn:
        model_obj = results[pred_model]['model']
        feature_order = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
        features = np.array([[input_vals[f] for f in feature_order]])
        pred  = model_obj.predict(features)[0]
        proba = model_obj.predict_proba(features)[0]
        fraud_prob = proba[1]
        legit_prob = proba[0]

        # Log prediction
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'model': pred_model,
            'amount': input_vals['Amount'],
            'result': 'FRAUD' if pred==1 else 'GENUINE',
            'confidence': f'{max(fraud_prob, legit_prob)*100:.1f}%'
        }
        st.session_state['pred_log'].insert(0, log_entry)
        st.session_state['pred_log'] = st.session_state['pred_log'][:10]

        # ── Result card ───────────────────────────────────────────────────
        col_res, col_chart = st.columns([1, 1])
        with col_res:
            if pred == 1:
                st.markdown(f"""
                <div class="pred-fraud">
                    <div class="pred-icon">🚨</div>
                    <div class="pred-label">Alert — Fraud Detected</div>
                    <div class="pred-title">FRAUDULENT</div>
                    <p class="pred-body">
                        This transaction has been flagged as <b style="color:#fca5a5">fraudulent</b>
                        with <b style="color:#fca5a5">{fraud_prob*100:.1f}%</b> probability.<br>
                        Amount: <b style="color:#fca5a5">${input_vals['Amount']:,.2f}</b> · Model: {pred_model}
                    </p>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-label"><span>Fraud Probability</span><span>{fraud_prob*100:.1f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill-red" style="width:{fraud_prob*100:.1f}%"></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-legit">
                    <div class="pred-icon">✅</div>
                    <div class="pred-label">Safe — Genuine Transaction</div>
                    <div class="pred-title">LEGITIMATE</div>
                    <p class="pred-body">
                        This transaction appears <b style="color:#6ee7b7">genuine</b>
                        with <b style="color:#6ee7b7">{legit_prob*100:.1f}%</b> confidence.<br>
                        Amount: <b style="color:#6ee7b7">${input_vals['Amount']:,.2f}</b> · Model: {pred_model}
                    </p>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-label"><span>Legit Probability</span><span>{legit_prob*100:.1f}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill-green" style="width:{legit_prob*100:.1f}%"></div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_chart:
            st.markdown('<div class="panel" style="margin-top:0"><div class="panel-title"><span class="dot"></span>Probability Breakdown</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                number=dict(suffix='%', font=dict(color='#e8eaff', size=36,
                                                    family='JetBrains Mono')),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont=dict(color='#3d4270')),
                    bar=dict(color='#ef4444' if pred==1 else '#10b981', thickness=0.3),
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0,30],  color='rgba(16,185,129,0.1)'),
                        dict(range=[30,60], color='rgba(245,158,11,0.1)'),
                        dict(range=[60,100],color='rgba(239,68,68,0.1)'),
                    ],
                    threshold=dict(line=dict(color='white', width=2), thickness=0.75, value=50)
                ),
                title=dict(text='Fraud Probability', font=dict(color='#4b5280', size=13))
            ))
            fig_gauge.update_layout(**BASE_LAYOUT, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Download report ───────────────────────────────────────────────
        st.markdown('<div class="section-header"><div class="line"></div><div class="text">Prediction Report</div><div class="line"></div></div>', unsafe_allow_html=True)
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model': pred_model,
            'prediction': 'FRAUD' if pred==1 else 'GENUINE',
            'fraud_probability': round(float(fraud_prob), 6),
            'legit_probability': round(float(legit_prob), 6),
            'transaction_amount': input_vals['Amount'],
            'transaction_time':   input_vals['Time'],
            'features': {k: round(float(v), 6) for k, v in input_vals.items()}
        }
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥  Download Prediction Report (JSON)",
            data=report_json,
            file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime='application/json'
        )

    # ── Prediction log ────────────────────────────────────────────────────────
    if st.session_state['pred_log']:
        st.markdown('<div class="section-header"><div class="line"></div><div class="text">Recent Predictions</div><div class="line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span>Last 10 Predictions</div>', unsafe_allow_html=True)
        log_df = pd.DataFrame(st.session_state['pred_log'])
        st.dataframe(log_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    if df is None: st.warning("Upload dataset first."); st.stop()

    st.markdown("""
    <div class="panel">
        <div class="panel-title"><span class="dot"></span>Interactive Dataset Explorer</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: class_filter = st.selectbox("Class", ['All', 'Genuine (0)', 'Fraud (1)'])
    with col2: amt_min = st.number_input("Min Amount", value=0.0)
    with col3: amt_max = st.number_input("Max Amount", value=float(df['Amount'].max()))
    with col4: n_rows  = st.slider("Rows to show", 10, 200, 50)

    filtered = df.copy()
    if class_filter == 'Genuine (0)': filtered = filtered[filtered.Class==0]
    elif class_filter == 'Fraud (1)': filtered = filtered[filtered.Class==1]
    filtered = filtered[(filtered['Amount'] >= amt_min) & (filtered['Amount'] <= amt_max)]

    st.markdown(f"<p style='color:#4b5280;font-size:0.82rem'>Showing {min(n_rows, len(filtered)):,} of {len(filtered):,} matching rows</p>", unsafe_allow_html=True)
    st.dataframe(filtered.head(n_rows), use_container_width=True)

    # Download
    csv_buf = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥  Export Filtered Data (CSV)", data=csv_buf,
                       file_name="filtered_transactions.csv", mime='text/csv')

    # Missing values
    st.markdown('<div class="panel" style="margin-top:20px"><div class="panel-title"><span class="dot"></span>Data Quality Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing Values per Column**")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Feature', 'Missing']
        missing['%'] = (missing['Missing'] / len(df) * 100).round(4)
        st.dataframe(missing, use_container_width=True)
    with c2:
        st.markdown("**Data Types**")
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ['Feature', 'Type']
        st.dataframe(dtypes_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER (all pages)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-left">
        <div class="footer-brand">🛡️ <span>FraudShield</span> AI</div>
        <div class="footer-copy">© 2025 · Credit Card Fraud Detection System · All rights reserved</div>
        <div class="footer-stack" style="margin-top:6px">
            Streamlit · Pandas · NumPy · Scikit-learn · XGBoost · Plotly
        </div>
    </div>
    <div class="footer-links">
        <a class="footer-link" href="https://github.com/Pratham2474" target="_blank">⭐ GitHub</a>
        <a class="footer-link" href="https://www.linkedin.com/in/prathamesh-nehete-8b27b8249" target="_blank">💼 LinkedIn</a>
        <a class="footer-link" href="https://kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">📦 Dataset</a>
    </div>
</div>
""", unsafe_allow_html=True)