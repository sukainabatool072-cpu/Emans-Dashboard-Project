import streamlit as st
import pandas as pd
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
import seaborn as sns
from charts import (pie_chart, histogram, line_chart, bar_chart,
                    scatter_plot, box_plot, heatmap, area_chart,
                    count_plot, violin_plot)

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Fine Foods Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: ##e8f4f8; }
    [data-testid="stSidebar"] { background-color: #e8f5e9; }
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #FF6B35;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #8A9BB0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1A2634, #243447);
        border: 1px solid #2E3F50;
        border-left: 4px solid #FF6B35;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 900;
        color: #FF6B35;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #8A9BB0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        color: #5d3a1a;
        font-size: 1.2rem;
        font-weight: 700;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stMetric label { color: #8A9BB0 !important; }
    .stMetric div { color: #FF6B35 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loader ────────────────────────────────────────────────
@st.cache_data(show_spinner="🍕 Loading Fine Foods dataset...")
def load_data():
    DATA_PATH = "data/finefoods.csv"
    df = pd.read_csv(DATA_PATH)
    with gzip.open(DATA_PATH, 'rt', encoding='utf-8', errors='ignore') as f:
        current = {}
        for line in f:
            line = line.strip()
            if line == '':
                if current:
                    records.append(current)
                    current = {}
            elif ': ' in line:
                key, val = line.split(': ', 1)
                current[key] = val
        if current:
            records.append(current)

    df = pd.DataFrame(records)
    df.columns = df.columns.str.replace('product/', '', regex=False).str.replace('review/', '', regex=False)

    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['date'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    def parse_help(h):
        try:
            n, d = h.split('/')
            n, d = int(n), int(d)
            return n / d if d > 0 else 0
        except:
            return 0

    df['helpfulness_ratio'] = df['helpfulness'].apply(parse_help)
    df['helpful_votes'] = df['helpfulness'].apply(
        lambda h: int(str(h).split('/')[0]) if '/' in str(h) else 0)
    df['total_votes'] = df['helpfulness'].apply(
        lambda h: int(str(h).split('/')[1]) if '/' in str(h) else 0)

    df['review_length'] = df['text'].fillna('').apply(len)
    df['word_count'] = df['text'].fillna('').apply(lambda x: len(x.split()))
    df['sentiment'] = df['score'].apply(
        lambda s: 'Positive' if s >= 4 else ('Neutral' if s == 3 else 'Negative'))

    df = df.dropna(subset=['score', 'date'])
    return df


# ── Load Data ──────────────────────────────────────────────────
df_full = load_data()

# ── Sidebar Filters ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Filters")
    st.markdown("---")

    # Score Range
    st.markdown("**⭐ Score Range**")
    score_range = st.slider("", 1, 5, (1, 5), key="score")

    # Year Range
    min_y, max_y = int(df_full['year'].min()), int(df_full['year'].max())
    st.markdown("**📅 Year Range**")
    year_range = st.slider("", min_y, max_y, (min_y, max_y), key="year")

    # Sentiment
    st.markdown("**😊 Sentiment**")
    sentiment = st.selectbox("", ['All', 'Positive', 'Neutral', 'Negative'])

    # Multi-select scores
    st.markdown("**🎯 Specific Scores**")
    selected_scores = st.multiselect("Select star ratings",
                                      [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    # Text search
    st.markdown("**🔍 Search Reviews**")
    search_text = st.text_input("Keyword in review/summary", "")

    # Reset
    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("<small style='color:#8A9BB0'>Amazon Fine Foods Reviews<br>EDA Dashboard Project</small>",
                unsafe_allow_html=True)


# ── Apply Filters ──────────────────────────────────────────────
df = df_full.copy()
df = df[(df['score'] >= score_range[0]) & (df['score'] <= score_range[1])]
df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if sentiment != 'All':
    df = df[df['sentiment'] == sentiment]
if selected_scores:
    df = df[df['score'].isin(selected_scores)]
if search_text:
    mask = (df['summary'].fillna('').str.contains(search_text, case=False) |
            df['text'].fillna('').str.contains(search_text, case=False))
    df = df[mask]


# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🍽️ Amazon Fine Foods Reviews</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Data Analysis Dashboard · EDA Course Project</div>',
            unsafe_allow_html=True)

if len(df) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.stop()


# ── KPI Cards ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Key Metrics</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Total Reviews</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["score"].mean():.2f}⭐</div><div class="kpi-label">Avg Score</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["productId"].nunique():,}</div><div class="kpi-label">Unique Products</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df["userId"].nunique():,}</div><div class="kpi-label">Unique Users</div></div>', unsafe_allow_html=True)
with k5:
    pos_pct = (df['sentiment'] == 'Positive').mean() * 100
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{pos_pct:.1f}%</div><div class="kpi-label">Positive Reviews</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Row 1: Distribution Charts ─────────────────────────────────
st.markdown('<div class="section-header">📊 Score & Sentiment Distribution</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(pie_chart(df))
with col2:
    st.pyplot(count_plot(df))
with col3:
    st.pyplot(histogram(df))

# ── Row 2: Trends Over Time ────────────────────────────────────
st.markdown('<div class="section-header">📅 Trends Over Time</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.pyplot(line_chart(df))
with col5:
    st.pyplot(area_chart(df))

# ── Row 3: Category & Relationship ────────────────────────────
st.markdown('<div class="section-header">🔍 Score Analysis</div>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    st.pyplot(bar_chart(df))
with col7:
    st.pyplot(scatter_plot(df))

# ── Row 4: Statistical Charts ─────────────────────────────────
st.markdown('<div class="section-header">📈 Statistical Distributions</div>', unsafe_allow_html=True)
col8, col9 = st.columns(2)
with col8:
    st.pyplot(box_plot(df))
with col9:
    st.pyplot(violin_plot(df))

# ── Row 5: Heatmap ─────────────────────────────────────────────
st.markdown('<div class="section-header">🔥 Correlation Analysis</div>', unsafe_allow_html=True)
col10, col11 = st.columns([2, 1])
with col10:
    st.pyplot(heatmap(df))
with col11:
    st.markdown("""
    <div style='background:#1A2634; border-radius:10px; padding:1rem; color:#5d3a1a; font-size:0.9rem;'>
    <b style='color:#FF6B35'>📋 What this shows:</b><br><br>
    The heatmap shows how different numerical features are correlated.<br><br>
    🟢 <b>Green/Red</b> = Strong correlation<br>
    ⚪ <b>Near 0</b> = Weak correlation<br><br>
    <b>Key insight:</b> Review length and word count are highly correlated, while score has weak correlation with review length.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4A5568; font-size:0.8rem; padding:1rem'>
    Amazon Fine Foods Reviews Dashboard · Built with Streamlit, Pandas, Matplotlib & Seaborn
</div>
""", unsafe_allow_html=True)
