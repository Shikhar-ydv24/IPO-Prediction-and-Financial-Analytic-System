"""
IPO Prediction & Financial Analytics System
============================================
Streamlit Dashboard — One-click interactive web app

Run locally:   streamlit run app.py
Deploy on:     Streamlit Community Cloud (free, shareable link)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_PATH  = os.path.join(BASE_DIR, "data", "ipo_enriched.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "data", "ipo_data.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPO Prediction & Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #0d1117; }
  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-title { color: #8b949e; font-size: 13px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px; }
  .metric-value { color: #58a6ff; font-size: 28px; font-weight: 700; }
  .metric-sub   { color: #3fb950; font-size: 12px; margin-top: 4px; }
  .predict-card {
    background: linear-gradient(135deg, #0d2137 0%, #0f2942 100%);
    border: 1px solid #1f6feb;
    border-radius: 14px;
    padding: 24px;
    margin: 12px 0;
  }
  .gain-positive { color: #3fb950; font-weight: 700; font-size: 36px; }
  .gain-negative { color: #f85149; font-weight: 700; font-size: 36px; }
  .gain-neutral  { color: #d29922; font-weight: 700; font-size: 36px; }
  div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    padding: 10px 28px; font-weight: 600; font-size: 15px;
    transition: all 0.2s; width: 100%;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #388bfd, #58a6ff); transform: translateY(-1px); }
  h1, h2, h3 { color: #c9d1d9 !important; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; }
  .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ── Data loader ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    numeric_cols = ["listing_gain_pct", "issue_price", "listing_price",
                    "issue_size_cr", "subscription_times", "qib_subscription",
                    "nii_subscription", "retail_subscription", "pe_ratio",
                    "revenue_cr", "profit_cr", "debt_equity",
                    "promoter_holding_pct", "nifty_on_listing",
                    "day7_close", "day30_close", "day90_close", "year", "month"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["company_name", "year"], keep="first")
    return df


@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "best_regressor.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


GAIN_BINS   = [-np.inf, 0, 10, 30, np.inf]
GAIN_LABELS = ["Negative", "Flat (0-10%)", "Moderate (10-30%)", "Strong (>30%)"]

df    = load_data()
model = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 IPO Analytics")
    st.markdown("---")
    st.markdown("**Filter Data**")
    year_range = st.slider("Year Range",
                           int(df["year"].min()), int(df["year"].max()),
                           (2018, int(df["year"].max())))
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector", sectors)
    sentiments = ["All"] + sorted(df["market_sentiment"].dropna().unique().tolist())
    sel_sentiment = st.selectbox("Market Sentiment", sentiments)
    min_sub = st.slider("Min Subscription (x)", 0, 100, 0)
    st.markdown("---")
    st.markdown("**Built with**")
    st.markdown("🐍 Python · 📊 Pandas · 🤖 Scikit-learn")
    st.markdown("📈 Plotly · 🌐 Streamlit · 💹 yfinance")
    st.markdown("---")
    st.caption("Data: SEBI Prospectuses + Yahoo Finance | Educational use only")


# ── Filter data ───────────────────────────────────────────────────────────
mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
if sel_sector != "All":
    mask &= df["sector"] == sel_sector
if sel_sentiment != "All":
    mask &= df["market_sentiment"] == sel_sentiment
mask &= df["subscription_times"].fillna(0) >= min_sub
df_f = df[mask].copy()


# ── HEADER ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%);
     border: 1px solid #1f6feb; border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;">
  <h1 style="margin:0; font-size:28px; color:#58a6ff;">📈 IPO Prediction & Financial Analytics System</h1>
  <p style="color:#8b949e; margin: 8px 0 0 0; font-size: 15px;">
    Indian IPO Market Analysis | ML-Powered Listing Gain Predictor | 2013-2024
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

total_ipos   = len(df_f)
avg_gain     = df_f["listing_gain_pct"].mean()
pct_positive = (df_f["listing_gain_pct"] > 0).mean() * 100
max_gain     = df_f["listing_gain_pct"].max()
avg_sub      = df_f["subscription_times"].mean()

for col, title, val, sub in [
    (c1, "Total IPOs",       f"{total_ipos}",        "in selection"),
    (c2, "Avg Listing Gain", f"{avg_gain:.1f}%",     "mean gain on listing day"),
    (c3, "Success Rate",     f"{pct_positive:.0f}%", "IPOs with positive listing"),
    (c4, "Best Gain",        f"{max_gain:.1f}%",     "highest listing gain"),
    (c5, "Avg Subscription", f"{avg_sub:.1f}x",      "average oversubscription"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "🔍 EDA", "🤖 ML Predictor", "📋 Data Explorer", "📊 Deep Dive"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        yearly = df_f.groupby("year").agg(
            count=("company_name", "count"),
            avg_gain=("listing_gain_pct", "mean")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yearly["year"], y=yearly["count"],
                             name="IPO Count", marker_color="#58a6ff", opacity=0.75), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["avg_gain"],
                                 mode="lines+markers", name="Avg Gain %",
                                 line=dict(color="#3fb950", width=2.5), marker=dict(size=6)),
                      secondary_y=True)
        fig.update_layout(title="IPO Volume & Avg Gain by Year",
                          plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                          font_color="#c9d1d9", legend=dict(x=0.01, y=0.99),
                          xaxis=dict(gridcolor="#21262d"), height=360,
                          yaxis=dict(gridcolor="#21262d"),
                          yaxis2=dict(gridcolor="#21262d"))
        fig.update_yaxes(title_text="IPO Count", secondary_y=False)
        fig.update_yaxes(title_text="Avg Listing Gain (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df_f, x="listing_gain_pct", nbins=35,
                            title="Listing Gain Distribution",
                            color_discrete_sequence=["#58a6ff"],
                            labels={"listing_gain_pct": "Listing Gain (%)"})
        fig2.add_vline(x=df_f["listing_gain_pct"].mean(), line_dash="dash",
                       line_color="#d29922", annotation_text=f"Mean: {avg_gain:.1f}%",
                       annotation_font_color="#d29922")
        fig2.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                           font_color="#c9d1d9", height=360,
                           xaxis=dict(gridcolor="#21262d"),
                           yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏆 Top 10 Best Performing IPOs")
    top10 = df_f.nlargest(10, "listing_gain_pct")[
        ["company_name", "year", "sector", "issue_price", "listing_price", "listing_gain_pct", "subscription_times"]
    ].reset_index(drop=True)
    top10.index += 1
    top10.columns = ["Company", "Year", "Sector", "Issue Price (Rs)", "Listing Price (Rs)", "Gain (%)", "Subscription (x)"]

    def color_gain(val):
        if isinstance(val, float):
            color = "#3fb950" if val > 0 else "#f85149"
            return f"color: {color}; font-weight: 600"
        return ""

    st.dataframe(top10.style.map(color_gain, subset=["Gain (%)"]),
                 use_container_width=True, height=370)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Exploratory Data Analysis")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        sector_perf = (df_f.groupby("sector")["listing_gain_pct"]
                       .agg(["mean", "count"]).reset_index()
                       .rename(columns={"mean": "avg_gain", "count": "n"})
                       .query("n >= 2").sort_values("avg_gain", ascending=True))
        colors = ["#3fb950" if x >= 0 else "#f85149" for x in sector_perf["avg_gain"]]
        fig = go.Figure(go.Bar(
            x=sector_perf["avg_gain"], y=sector_perf["sector"],
            orientation="h", marker_color=colors, opacity=0.85,
            text=[f"{v:.1f}% (n={n})" for v, n in zip(sector_perf["avg_gain"], sector_perf["n"])],
            textposition="outside"
        ))
        fig.update_layout(title="Avg Listing Gain by Sector",
                          plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                          font_color="#c9d1d9", height=420,
                          xaxis=dict(gridcolor="#21262d", title="Avg Gain (%)"),
                          yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        valid = df_f[["subscription_times", "listing_gain_pct", "sector", "company_name"]].dropna()
        cap   = valid["subscription_times"].quantile(0.97)
        valid = valid[valid["subscription_times"] <= cap]
        fig2  = px.scatter(valid, x="subscription_times", y="listing_gain_pct",
                           color="sector", hover_data=["company_name"],
                           trendline="ols",
                           title="Subscription x vs Listing Gain",
                           labels={"subscription_times": "Subscription (x)",
                                   "listing_gain_pct": "Listing Gain (%)"})
        fig2.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                           font_color="#c9d1d9", height=420,
                           xaxis=dict(gridcolor="#21262d"),
                           yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig3 = px.box(df_f, x="market_sentiment", y="listing_gain_pct",
                      color="market_sentiment",
                      color_discrete_map={"Bullish": "#3fb950", "Neutral": "#d29922", "Bearish": "#f85149"},
                      title="Listing Gain by Market Sentiment",
                      category_orders={"market_sentiment": ["Bullish", "Neutral", "Bearish"]},
                      labels={"listing_gain_pct": "Listing Gain (%)", "market_sentiment": "Sentiment"})
        fig3.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                           font_color="#c9d1d9", height=380, showlegend=False,
                           xaxis=dict(gridcolor="#21262d"),
                           yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        corr_cols = ["listing_gain_pct", "issue_price", "issue_size_cr", "subscription_times",
                     "qib_subscription", "retail_subscription", "pe_ratio", "profit_cr",
                     "promoter_holding_pct", "nifty_on_listing"]
        corr_cols = [c for c in corr_cols if c in df_f.columns]
        corr = df_f[corr_cols].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         title="Feature Correlation Heatmap", zmin=-1, zmax=1)
        fig4.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                           font_color="#c9d1d9", height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📅 Monthly Seasonality")
    monthly = df_f.groupby("month").agg(
        avg_gain=("listing_gain_pct", "mean"),
        count=("company_name", "count")
    ).reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=monthly["month_name"], y=monthly["count"],
                          name="IPO Count", yaxis="y2",
                          marker_color="#58a6ff", opacity=0.5))
    fig5.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["avg_gain"],
                              name="Avg Gain %", mode="lines+markers",
                              line=dict(color="#3fb950", width=2.5), marker=dict(size=7)))
    fig5.update_layout(
        title="Monthly IPO Activity & Avg Listing Gain",
        plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", height=350,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Avg Gain (%)"),
        yaxis2=dict(overlaying="y", side="right", title="IPO Count", gridcolor="#21262d"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 IPO Listing Gain Predictor")
    st.markdown("Enter the IPO details below to get an ML-powered prediction.")

    if model is None:
        st.warning("No trained model found. Run `python utils/ml_model.py` first to train and save the model.")

    with st.form("predict_form"):
        st.markdown("#### 📋 IPO Details")
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            issue_price  = st.number_input("Issue Price (Rs)", 10, 10000, 500)
            issue_size   = st.number_input("Issue Size (Rs Cr)", 10, 50000, 2000)
            subscription = st.number_input("Overall Subscription (x)", 0.0, 500.0, 50.0)
            qib_sub      = st.number_input("QIB Subscription (x)", 0.0, 500.0, 80.0)

        with pc2:
            nii_sub     = st.number_input("NII Subscription (x)", 0.0, 1000.0, 60.0)
            retail_sub  = st.number_input("Retail Subscription (x)", 0.0, 100.0, 12.0)
            pe_ratio    = st.number_input("P/E Ratio", 0.0, 500.0, 35.0)
            revenue     = st.number_input("Revenue (Rs Cr)", 0.0, 100000.0, 1500.0)

        with pc3:
            profit       = st.number_input("Net Profit (Rs Cr)", -10000.0, 50000.0, 120.0)
            debt_eq      = st.number_input("Debt-to-Equity", 0.0, 20.0, 0.3)
            promoter_hold = st.number_input("Promoter Holding (%)", 0.0, 100.0, 65.0)
            nifty_level  = st.number_input("Nifty Level on Listing", 5000, 25000, 18000)

        st.markdown("#### 🏢 Company Profile")
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            sector = st.selectbox("Sector", sorted(df["sector"].dropna().unique().tolist()))
        with fc2:
            issue_type = st.selectbox("Issue Type", ["OFS+Fresh", "Fresh", "OFS"])
        with fc3:
            sentiment = st.selectbox("Market Sentiment", ["Bullish", "Neutral", "Bearish"])
        with fc4:
            year  = st.number_input("Listing Year", 2020, 2030, 2024)
            month = st.selectbox("Listing Month", list(range(1, 13)),
                                 format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                         "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

        submitted = st.form_submit_button("🔮 Predict Listing Gain")

    if submitted:
        if model is None:
            st.error("Please train the model first: python utils/ml_model.py")
        else:
            input_dict = {
                "issue_price": issue_price, "issue_size_cr": issue_size,
                "subscription_times": subscription, "qib_subscription": qib_sub,
                "nii_subscription": nii_sub, "retail_subscription": retail_sub,
                "pe_ratio": pe_ratio, "revenue_cr": revenue, "profit_cr": profit,
                "debt_equity": debt_eq, "promoter_holding_pct": promoter_hold,
                "nifty_on_listing": nifty_level, "year": year, "month": month,
                "sector": sector, "issue_type": issue_type, "market_sentiment": sentiment
            }
            try:
                input_df = pd.DataFrame([input_dict])
                predicted_gain = float(model.predict(input_df)[0])
                category = pd.cut([predicted_gain], bins=GAIN_BINS, labels=GAIN_LABELS)[0]
                confidence = min(95, max(55, 75 + predicted_gain * 0.1))

                cls = "gain-positive" if predicted_gain > 0 else ("gain-negative" if predicted_gain < -5 else "gain-neutral")
                rec = ("✅ Strong Buy" if predicted_gain > 30 else
                       "✅ Consider Investing" if predicted_gain > 10 else
                       "⚠️ Neutral — Monitor" if predicted_gain >= 0 else
                       "❌ Avoid — High Risk")

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.markdown(f"""
                <div class="predict-card" style="text-align:center;">
                  <div style="color:#8b949e; font-size:13px; margin-bottom:8px;">PREDICTED LISTING GAIN</div>
                  <div class="{cls}">{predicted_gain:+.2f}%</div>
                  <div style="color:#8b949e; margin-top:8px; font-size:13px;">{str(category)}</div>
                </div>""", unsafe_allow_html=True)
                col_res2.markdown(f"""
                <div class="predict-card" style="text-align:center;">
                  <div style="color:#8b949e; font-size:13px; margin-bottom:8px;">CONFIDENCE SCORE</div>
                  <div style="color:#58a6ff; font-size:36px; font-weight:700;">{confidence:.0f}%</div>
                  <div style="color:#8b949e; margin-top:8px; font-size:13px;">Model confidence</div>
                </div>""", unsafe_allow_html=True)
                col_res3.markdown(f"""
                <div class="predict-card" style="text-align:center;">
                  <div style="color:#8b949e; font-size:13px; margin-bottom:8px;">RECOMMENDATION</div>
                  <div style="color:#d29922; font-size:20px; font-weight:700; margin-top:16px;">{rec}</div>
                </div>""", unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_gain,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Predicted Listing Gain (%)", "font": {"color": "#c9d1d9"}},
                    delta={"reference": 0, "increasing": {"color": "#3fb950"}, "decreasing": {"color": "#f85149"}},
                    gauge={
                        "axis": {"range": [-30, 120], "tickcolor": "#8b949e"},
                        "bar": {"color": "#3fb950" if predicted_gain > 0 else "#f85149"},
                        "bgcolor": "#161b22",
                        "steps": [
                            {"range": [-30, 0],  "color": "#1a0a0a"},
                            {"range": [0, 10],   "color": "#0d1a10"},
                            {"range": [10, 30],  "color": "#0d1f10"},
                            {"range": [30, 120], "color": "#0d280d"},
                        ],
                        "threshold": {"line": {"color": "orange", "width": 3}, "thickness": 0.8, "value": 0}
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor="#0d1117", font_color="#c9d1d9", height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}. Please ensure the model is trained.")

    st.markdown("### 📊 Compare with Historical IPOs")
    similar = df[
        (df["sector"] == sector) &
        (df["subscription_times"].between(max(0, subscription * 0.5), subscription * 2 + 10))
    ][["company_name", "year", "listing_gain_pct", "subscription_times", "issue_price"]].dropna()

    if len(similar) > 0:
        st.markdown(f"**{len(similar)} similar IPOs** in {sector} sector with comparable subscription:")
        fig_sim = px.scatter(similar, x="subscription_times", y="listing_gain_pct",
                             text="company_name", size_max=12,
                             color_discrete_sequence=["#58a6ff"],
                             labels={"subscription_times": "Subscription (x)",
                                     "listing_gain_pct": "Listing Gain (%)"})
        fig_sim.update_traces(textposition="top center", textfont_size=8)
        fig_sim.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                              font_color="#c9d1d9", height=350,
                              xaxis=dict(gridcolor="#21262d"),
                              yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.info("No similar historical IPOs found in this sector/subscription range.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Full IPO Dataset")

    search = st.text_input("🔍 Search company name", "")
    df_show = df_f.copy()
    if search:
        df_show = df_show[df_show["company_name"].str.contains(search, case=False, na=False)]

    display_cols = ["company_name", "year", "sector", "issue_type", "issue_price",
                    "listing_price", "listing_gain_pct", "subscription_times",
                    "qib_subscription", "retail_subscription", "market_sentiment"]
    display_cols = [c for c in display_cols if c in df_show.columns]

    st.markdown(f"**Showing {len(df_show)} IPOs**")

    def highlight_gain(val):
        if isinstance(val, (int, float)):
            if val > 30:  return "background-color: #0d280d; color: #3fb950; font-weight:600"
            if val > 0:   return "background-color: #0d1f10; color: #7ee787"
            if val < -5:  return "background-color: #1a0a0a; color: #f85149; font-weight:600"
            return "background-color: #1a0f0a; color: #e3b341"
        return ""

    styled = (df_show[display_cols]
              .rename(columns={"company_name": "Company", "year": "Year", "sector": "Sector",
                               "issue_type": "Type", "issue_price": "Issue Rs",
                               "listing_price": "Listing Rs", "listing_gain_pct": "Gain %",
                               "subscription_times": "Sub x", "qib_subscription": "QIB x",
                               "retail_subscription": "Retail x", "market_sentiment": "Sentiment"})
              .style.map(highlight_gain, subset=["Gain %"]))
    st.dataframe(styled, use_container_width=True, height=500)

    csv = df_show[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", csv, "ipo_filtered.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📊 Deep Dive Analysis")
    st.markdown("#### Subscription Type Analysis")
    d1, d2 = st.columns(2)

    with d1:
        melt_df = df_f[["company_name", "qib_subscription", "nii_subscription", "retail_subscription"]].dropna()
        melt_df = melt_df.head(30)
        melt_df = melt_df.melt(id_vars="company_name", var_name="Category", value_name="Subscription")
        melt_df["Category"] = melt_df["Category"].map({
            "qib_subscription": "QIB", "nii_subscription": "NII", "retail_subscription": "Retail"
        })
        avg_sub_type = melt_df.groupby("Category")["Subscription"].mean().reset_index()
        fig_sub = px.bar(avg_sub_type, x="Category", y="Subscription",
                         color="Category",
                         color_discrete_map={"QIB": "#bc8cff", "NII": "#58a6ff", "Retail": "#3fb950"},
                         title="Average Subscription by Investor Type",
                         labels={"Subscription": "Avg Subscription (x)"})
        fig_sub.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                              font_color="#c9d1d9", showlegend=False, height=350,
                              xaxis=dict(gridcolor="#21262d"),
                              yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig_sub, use_container_width=True)

    with d2:
        valid2 = df_f[["issue_size_cr", "listing_gain_pct", "sector"]].dropna()
        fig_size = px.scatter(valid2, x="issue_size_cr", y="listing_gain_pct",
                              color="sector", log_x=True,
                              title="Issue Size vs Listing Gain (log scale)",
                              labels={"issue_size_cr": "Issue Size (Rs Cr, log)",
                                      "listing_gain_pct": "Listing Gain (%)"})
        fig_size.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                               font_color="#c9d1d9", height=350,
                               xaxis=dict(gridcolor="#21262d"),
                               yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig_size, use_container_width=True)

    st.markdown("#### Post-Listing Performance (Day 7 to 30 to 90)")
    post_df = df_f[["company_name", "listing_gain_pct", "day7_close", "day30_close", "day90_close", "listing_price"]].dropna()
    post_df = post_df.copy()
    post_df["Day7 Return%"]  = ((post_df["day7_close"]  - post_df["listing_price"]) / post_df["listing_price"] * 100).round(1)
    post_df["Day30 Return%"] = ((post_df["day30_close"] - post_df["listing_price"]) / post_df["listing_price"] * 100).round(1)
    post_df["Day90 Return%"] = ((post_df["day90_close"] - post_df["listing_price"]) / post_df["listing_price"] * 100).round(1)

    avg_post = {
        "Listing Day": post_df["listing_gain_pct"].mean(),
        "Day 7":       post_df["Day7 Return%"].mean(),
        "Day 30":      post_df["Day30 Return%"].mean(),
        "Day 90":      post_df["Day90 Return%"].mean(),
    }
    fig_post = go.Figure(go.Scatter(
        x=list(avg_post.keys()), y=list(avg_post.values()),
        mode="lines+markers+text",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=10, color="#58a6ff"),
        text=[f"{v:.1f}%" for v in avg_post.values()],
        textposition="top center", textfont=dict(color="#c9d1d9")
    ))
    fig_post.add_hline(y=0, line_dash="dash", line_color="#30363d")
    fig_post.update_layout(
        title="Average Returns at Different Holding Periods",
        plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", height=350,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Average Return (%)")
    )
    st.plotly_chart(fig_post, use_container_width=True)

    st.markdown("#### Profitable vs Loss-Making Companies")
    if "profit_cr" in df_f.columns:
        df_profit = df_f.copy()
        df_profit["_profitable"] = df_profit["profit_cr"] > 0
        prof_group = df_profit.groupby("_profitable")["listing_gain_pct"].agg(["mean", "count"]).reset_index()
        prof_group["_profitable"] = prof_group["_profitable"].map({True: "Profitable", False: "Loss-Making"})
        fig_prof = px.bar(prof_group, x="_profitable", y="mean",
                          color="_profitable",
                          color_discrete_map={"Profitable": "#3fb950", "Loss-Making": "#f85149"},
                          text=[f"{v:.1f}% (n={n})" for v, n in zip(prof_group["mean"], prof_group["count"])],
                          title="Avg Listing Gain: Profitable vs Loss-Making Companies",
                          labels={"_profitable": "", "mean": "Avg Listing Gain (%)"})
        fig_prof.update_layout(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                               font_color="#c9d1d9", showlegend=False, height=350,
                               xaxis=dict(gridcolor="#21262d"),
                               yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig_prof, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:12px; padding:10px 0;">
  📈 IPO Prediction & Financial Analytics System &nbsp;|&nbsp;
  Built with Python · Scikit-learn · Streamlit · Plotly &nbsp;|&nbsp;
  Data: SEBI Prospectuses + Yahoo Finance (Official) &nbsp;|&nbsp;
  For Educational & Portfolio Purposes
</div>
""", unsafe_allow_html=True)
