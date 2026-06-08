"""
Exploratory Data Analysis (EDA) Module
---------------------------------------
Comprehensive EDA for Indian IPO dataset:
  - Distribution analysis
  - Sector-wise performance
  - Subscription vs listing gain correlation
  - Market sentiment impact
  - Temporal trends
  - Feature correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi":       120,
})

ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
PINK     = "#ff7b72"

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "ipo_enriched.csv")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = DATA_PATH if os.path.exists(DATA_PATH) else os.path.join(BASE_DIR, "data", "ipo_data.csv")
    df   = pd.read_csv(path)
    # Ensure numeric types
    for col in ["listing_gain_pct", "subscription_times", "issue_size_cr",
                "qib_subscription", "nii_subscription", "retail_subscription",
                "pe_ratio", "profit_cr", "nifty_on_listing", "promoter_holding_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ────────────────────────────────────────────────────────────────────────────
# 1. Listing Gain Distribution
# ────────────────────────────────────────────────────────────────────────────
def plot_listing_gain_distribution(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IPO Listing Gain Distribution", fontsize=15, fontweight="bold", color="#c9d1d9")

    # Histogram
    ax = axes[0]
    gains = df["listing_gain_pct"].dropna()
    ax.hist(gains, bins=40, color=ACCENT, edgecolor="#0d1117", alpha=0.85)
    ax.axvline(gains.mean(),   color=ORANGE, lw=2, ls="--", label=f"Mean: {gains.mean():.1f}%")
    ax.axvline(gains.median(), color=GREEN,  lw=2, ls="--", label=f"Median: {gains.median():.1f}%")
    ax.set_xlabel("Listing Gain (%)")
    ax.set_ylabel("Number of IPOs")
    ax.set_title("Distribution of Listing Gains")
    ax.legend(framealpha=0.3)

    # Box plot by category
    ax2 = axes[1]
    bins   = [-np.inf, -5, 0, 10, 30, 60, np.inf]
    labels = ["< -5%", "-5–0%", "0–10%", "10–30%", "30–60%", "> 60%"]
    df["_cat"] = pd.cut(df["listing_gain_pct"], bins=bins, labels=labels)
    counts = df["_cat"].value_counts().reindex(labels)
    colors = [RED, PINK, ORANGE, ACCENT, GREEN, PURPLE]
    bars = ax2.bar(counts.index, counts.values, color=colors, edgecolor="#0d1117", alpha=0.88)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9, color="#c9d1d9")
    ax2.set_xlabel("Listing Gain Category")
    ax2.set_ylabel("Number of IPOs")
    ax2.set_title("IPOs by Listing Gain Category")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    df.drop(columns=["_cat"], inplace=True)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "01_listing_gain_distribution.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 2. Sector-wise Performance
# ────────────────────────────────────────────────────────────────────────────
def plot_sector_performance(df: pd.DataFrame, save: bool = True):
    sector_stats = (
        df.groupby("sector")["listing_gain_pct"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_gain", "median": "med_gain", "count": "ipo_count"})
        .query("ipo_count >= 2")
        .sort_values("avg_gain", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(13, max(6, len(sector_stats)*0.45)))
    fig.suptitle("Sector-wise Average IPO Listing Gain", fontsize=14, fontweight="bold", color="#c9d1d9")

    colors = [GREEN if x >= 0 else RED for x in sector_stats["avg_gain"]]
    bars   = ax.barh(sector_stats["sector"], sector_stats["avg_gain"],
                     color=colors, edgecolor="#0d1117", alpha=0.85, height=0.65)

    for bar, count in zip(bars, sector_stats["ipo_count"]):
        w = bar.get_width()
        ax.text(w + (0.5 if w >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                f"  n={count}", va="center", fontsize=7.5, color="#8b949e")

    ax.axvline(0, color="#30363d", lw=1.2)
    ax.set_xlabel("Average Listing Gain (%)")
    ax.set_title("Sectors with ≥ 2 IPOs")
    ax.grid(axis="x", alpha=0.4)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "02_sector_performance.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 3. Subscription vs Listing Gain
# ────────────────────────────────────────────────────────────────────────────
def plot_subscription_vs_gain(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Subscription Analysis vs Listing Gain", fontsize=14, fontweight="bold", color="#c9d1d9")

    sub_cols = [
        ("subscription_times", "Overall Subscription (×)", ACCENT),
        ("qib_subscription",   "QIB Subscription (×)",     PURPLE),
        ("retail_subscription","Retail Subscription (×)",   GREEN),
    ]

    for ax, (col, label, color) in zip(axes, sub_cols):
        valid = df[[col, "listing_gain_pct"]].dropna()
        # Cap at 99th percentile to handle outliers
        cap = valid[col].quantile(0.99)
        valid = valid[valid[col] <= cap]

        ax.scatter(valid[col], valid["listing_gain_pct"],
                   alpha=0.55, s=25, color=color, edgecolors="none")

        # Trend line
        z = np.polyfit(valid[col], valid["listing_gain_pct"], 1)
        p = np.poly1d(z)
        xline = np.linspace(valid[col].min(), valid[col].max(), 100)
        ax.plot(xline, p(xline), color=ORANGE, lw=2, ls="--", label="Trend")

        corr = valid[col].corr(valid["listing_gain_pct"])
        ax.set_title(f"{label}\n(r = {corr:.2f})", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("Listing Gain (%)" if ax == axes[0] else "")
        ax.axhline(0, color="#30363d", lw=1)
        ax.legend(fontsize=8, framealpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "03_subscription_vs_gain.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 4. Year-wise Trend
# ────────────────────────────────────────────────────────────────────────────
def plot_yearly_trends(df: pd.DataFrame, save: bool = True):
    yearly = (
        df.groupby("year")
        .agg(avg_gain=("listing_gain_pct","mean"),
             ipo_count=("company_name","count"),
             pct_positive=("listing_gain_pct", lambda x: (x > 0).mean() * 100))
        .reset_index()
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Year-wise IPO Market Trends", fontsize=14, fontweight="bold", color="#c9d1d9")

    # IPO count
    ax = axes[0]
    ax.bar(yearly["year"], yearly["ipo_count"], color=ACCENT, alpha=0.85, width=0.6)
    ax.set_ylabel("Number of IPOs")
    ax.set_title("IPO Volume by Year")

    # Avg gain
    ax2 = axes[1]
    colors2 = [GREEN if x >= 0 else RED for x in yearly["avg_gain"]]
    ax2.bar(yearly["year"], yearly["avg_gain"], color=colors2, alpha=0.85, width=0.6)
    ax2.axhline(0, color="#30363d", lw=1.2)
    ax2.set_ylabel("Avg Listing Gain (%)")
    ax2.set_title("Average Listing Gain by Year")

    # % positive
    ax3 = axes[2]
    ax3.plot(yearly["year"], yearly["pct_positive"], marker="o",
             color=GREEN, lw=2.5, ms=6, label="% Positive Listings")
    ax3.axhline(50, color=ORANGE, lw=1.5, ls="--", label="50% line")
    ax3.fill_between(yearly["year"], yearly["pct_positive"], 50,
                     where=yearly["pct_positive"] >= 50, alpha=0.15, color=GREEN)
    ax3.fill_between(yearly["year"], yearly["pct_positive"], 50,
                     where=yearly["pct_positive"] < 50,  alpha=0.15, color=RED)
    ax3.set_ylabel("% Positive Listings")
    ax3.set_xlabel("Year")
    ax3.set_title("% of IPOs with Positive Listing Gain")
    ax3.legend(fontsize=9, framealpha=0.3)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax3.set_ylim(0, 100)

    for ax in axes:
        ax.set_xticks(yearly["year"])
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "04_yearly_trends.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 5. Market Sentiment Impact
# ────────────────────────────────────────────────────────────────────────────
def plot_market_sentiment(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Market Sentiment vs IPO Performance", fontsize=14, fontweight="bold", color="#c9d1d9")

    sentiment_order = ["Bullish", "Neutral", "Bearish"]
    palette = {"Bullish": GREEN, "Neutral": ORANGE, "Bearish": RED}

    # Box plot
    ax = axes[0]
    groups = [df[df["market_sentiment"] == s]["listing_gain_pct"].dropna()
              for s in sentiment_order]
    bp = ax.boxplot(groups, patch_artist=True, labels=sentiment_order,
                    medianprops={"color": "white", "lw": 2})
    for patch, s in zip(bp["boxes"], sentiment_order):
        patch.set_facecolor(palette[s])
        patch.set_alpha(0.7)
    ax.axhline(0, color="#30363d", lw=1.2, ls="--")
    ax.set_ylabel("Listing Gain (%)")
    ax.set_title("Listing Gain by Market Sentiment")

    # Success rate bar
    ax2 = axes[1]
    stats = df.groupby("market_sentiment")["listing_gain_pct"].apply(
        lambda x: (x > 0).mean() * 100
    ).reindex(sentiment_order)
    bars = ax2.bar(stats.index, stats.values,
                   color=[palette[s] for s in sentiment_order],
                   alpha=0.85, width=0.5)
    for bar, val in zip(bars, stats.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", fontsize=11, color="#c9d1d9")
    ax2.set_ylabel("% Positive Listings")
    ax2.set_title("Success Rate by Market Sentiment")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "05_market_sentiment.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 6. Correlation Heatmap
# ────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    cols = [
        "listing_gain_pct", "issue_price", "issue_size_cr", "subscription_times",
        "qib_subscription", "nii_subscription", "retail_subscription",
        "pe_ratio", "profit_cr", "revenue_cr", "debt_equity",
        "promoter_holding_pct", "nifty_on_listing"
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7.5}, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold",
                 color="#c9d1d9", pad=15)
    ax.set_facecolor("#161b22")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "06_correlation_heatmap.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 7. Issue Type Performance
# ────────────────────────────────────────────────────────────────────────────
def plot_issue_type_analysis(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("IPO Issue Type Analysis", fontsize=14, fontweight="bold", color="#c9d1d9")

    type_stats = df.groupby("issue_type")["listing_gain_pct"].agg(["mean","count"]).reset_index()
    type_stats = type_stats.sort_values("mean", ascending=False)

    # Bar chart
    ax = axes[0]
    colors = [GREEN if x >= 0 else RED for x in type_stats["mean"]]
    bars = ax.bar(type_stats["issue_type"], type_stats["mean"],
                  color=colors, alpha=0.85, width=0.5)
    for bar, row in zip(bars, type_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{row.mean:.1f}%\n(n={row.count})", ha="center", fontsize=8.5, color="#c9d1d9")
    ax.set_ylabel("Avg Listing Gain (%)")
    ax.set_title("Average Gain by Issue Type")
    ax.axhline(0, color="#30363d", lw=1)

    # Pie: distribution
    ax2 = axes[1]
    pie_data = df["issue_type"].value_counts()
    colors_pie = [ACCENT, GREEN, PURPLE, ORANGE][:len(pie_data)]
    wedges, texts, autotexts = ax2.pie(
        pie_data.values, labels=pie_data.index, autopct="%1.0f%%",
        colors=colors_pie, startangle=140,
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
    )
    for t in texts + autotexts:
        t.set_color("#c9d1d9")
        t.set_fontsize(9)
    ax2.set_title("IPO Distribution by Issue Type")

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "07_issue_type_analysis.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# 8. Top & Bottom IPOs
# ────────────────────────────────────────────────────────────────────────────
def plot_top_bottom_ipos(df: pd.DataFrame, n: int = 10, save: bool = True):
    df_sorted = df[["company_name","listing_gain_pct","year","sector"]].dropna(subset=["listing_gain_pct"])
    top_n    = df_sorted.nlargest(n, "listing_gain_pct")
    bottom_n = df_sorted.nsmallest(n, "listing_gain_pct")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Top {n} & Bottom {n} IPOs by Listing Gain", fontsize=14, fontweight="bold", color="#c9d1d9")

    for ax, data, title, color in [
        (axes[0], top_n,    f"Top {n} IPOs (Best Gains)",   GREEN),
        (axes[1], bottom_n, f"Bottom {n} IPOs (Worst Losses)", RED)
    ]:
        bars = ax.barh(data["company_name"], data["listing_gain_pct"],
                       color=color, alpha=0.82, edgecolor="#0d1117")
        for bar, val in zip(bars, data["listing_gain_pct"]):
            ax.text(bar.get_width() + (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=8, color="#c9d1d9")
        ax.axvline(0, color="#30363d", lw=1.2)
        ax.set_xlabel("Listing Gain (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(ASSETS_DIR, "08_top_bottom_ipos.png")
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()
    plt.close()


# ────────────────────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────────────────────
def run_full_eda():
    print("=" * 60)
    print("RUNNING FULL EDA")
    print("=" * 60)
    df = load_data()
    print(f"Dataset: {df.shape[0]} IPOs × {df.shape[1]} features\n")

    print("Summary Statistics:")
    print(df["listing_gain_pct"].describe().round(2).to_string())
    print()

    plot_listing_gain_distribution(df)
    plot_sector_performance(df)
    plot_subscription_vs_gain(df)
    plot_yearly_trends(df)
    plot_market_sentiment(df)
    plot_correlation_heatmap(df)
    plot_issue_type_analysis(df)
    plot_top_bottom_ipos(df)

    print("\n✅ EDA complete. All charts saved to assets/")


if __name__ == "__main__":
    run_full_eda()
