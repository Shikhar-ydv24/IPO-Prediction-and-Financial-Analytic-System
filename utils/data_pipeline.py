"""
IPO Data Pipeline
-----------------
Fetches real post-listing stock data using yfinance (Yahoo Finance official API)
and enriches the base IPO dataset with live financial metrics.

Sources used:
  - data/ipo_data.csv  : Curated IPO metadata (issue price, subscription, sector, etc.)
                         compiled from SEBI prospectuses & NSE/BSE announcements
  - yfinance           : Official Yahoo Finance API — post-listing price history,
                         fundamentals (PE, market cap, revenue, EPS)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH  = os.path.join(BASE_DIR, "data", "ipo_data.csv")
ENRICHED_PATH  = os.path.join(BASE_DIR, "data", "ipo_enriched.csv")


# ---------------------------------------------------------------------------
# Helper: fetch live fundamentals for a ticker from Yahoo Finance
# ---------------------------------------------------------------------------
def fetch_yfinance_fundamentals(symbol: str) -> dict:
    """
    Pull live/latest fundamentals for an NSE-listed company via yfinance.
    NSE symbols on Yahoo Finance use the '.NS' suffix.
    """
    result = {}
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info   = ticker.info

        result["yf_market_cap_cr"]   = round(info.get("marketCap", 0) / 1e7, 2) if info.get("marketCap") else np.nan
        result["yf_pe_ratio"]        = info.get("trailingPE", np.nan)
        result["yf_52w_high"]        = info.get("fiftyTwoWeekHigh", np.nan)
        result["yf_52w_low"]         = info.get("fiftyTwoWeekLow", np.nan)
        result["yf_current_price"]   = info.get("currentPrice", np.nan)
        result["yf_beta"]            = info.get("beta", np.nan)
        result["yf_revenue_cr"]      = round(info.get("totalRevenue", 0) / 1e7, 2) if info.get("totalRevenue") else np.nan
        result["yf_profit_margin"]   = info.get("profitMargins", np.nan)
        result["yf_roe"]             = info.get("returnOnEquity", np.nan)
        result["yf_debt_to_equity"]  = info.get("debtToEquity", np.nan)
        result["yf_book_value"]      = info.get("bookValue", np.nan)
        result["yf_eps"]             = info.get("trailingEps", np.nan)

        logger.info(f"  ✓ Fetched fundamentals for {symbol}")
    except Exception as e:
        logger.warning(f"  ✗ Could not fetch {symbol}: {e}")
    return result


def fetch_listing_day_return(symbol: str, listing_year: int, listing_month: int) -> dict:
    """
    Use yfinance to compute actual Day-1 and Day-30 returns from listing date.
    Listing date is approximated to the 1st of the given month when exact date
    is not stored; adjust if you have exact listing dates.
    """
    result = {}
    try:
        # Approximate listing date
        listing_date = datetime(listing_year, listing_month, 1)
        start = listing_date - timedelta(days=5)
        end   = listing_date + timedelta(days=35)

        ticker = yf.Ticker(f"{symbol}.NS")
        hist   = ticker.history(start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            return result

        hist = hist.sort_index()
        open_price  = hist["Open"].iloc[0]
        close_day1  = hist["Close"].iloc[0]
        close_day30 = hist["Close"].iloc[-1] if len(hist) >= 20 else np.nan

        if open_price and open_price > 0:
            result["yf_day1_return_pct"]  = round((close_day1 - open_price) / open_price * 100, 2)
            result["yf_day30_return_pct"] = round((close_day30 - open_price) / open_price * 100, 2) if not np.isnan(close_day30) else np.nan

        result["yf_listing_open"]  = round(open_price, 2)
        result["yf_listing_close"] = round(close_day1, 2)
        result["yf_day30_close"]   = round(close_day30, 2) if not np.isnan(close_day30) else np.nan

        logger.info(f"  ✓ Fetched listing-day history for {symbol}")
    except Exception as e:
        logger.warning(f"  ✗ History failed for {symbol}: {e}")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(max_tickers: Optional[int] = None, skip_yfinance: bool = False) -> pd.DataFrame:
    """
    Load base CSV → optionally enrich with yfinance → save enriched CSV.

    Parameters
    ----------
    max_tickers  : Limit how many tickers to query (useful for quick testing).
    skip_yfinance: If True, skip network calls and just clean/feature-engineer the base CSV.
    """
    logger.info("=" * 60)
    logger.info("IPO DATA PIPELINE STARTED")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load base dataset
    # ------------------------------------------------------------------
    logger.info(f"[1/4] Loading base dataset from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"      Loaded {len(df)} IPO records")

    # Remove duplicate company entries (keep first occurrence)
    df = df.drop_duplicates(subset=["company_name", "year"], keep="first")
    logger.info(f"      After deduplication: {len(df)} records")

    # ------------------------------------------------------------------
    # Step 2: Clean & type-cast
    # ------------------------------------------------------------------
    logger.info("[2/4] Cleaning and type-casting columns")

    numeric_cols = [
        "issue_price", "listing_price", "listing_gain_pct", "issue_size_cr",
        "subscription_times", "qib_subscription", "nii_subscription",
        "retail_subscription", "year", "month", "nifty_on_listing",
        "pe_ratio", "revenue_cr", "profit_cr", "debt_equity",
        "promoter_holding_pct", "day7_close", "day30_close", "day90_close"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["issue_type"]       = df["issue_type"].astype("category")
    df["sector"]           = df["sector"].astype("category")
    df["market_sentiment"] = df["market_sentiment"].astype("category")

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("[3/4] Engineering derived features")

    # Oversubscription flag
    df["is_oversubscribed"]         = (df["subscription_times"] >= 1).astype(int)
    df["heavily_oversubscribed"]    = (df["subscription_times"] >= 50).astype(int)

    # Valuation & size
    df["log_issue_size"]            = np.log1p(df["issue_size_cr"])
    df["log_subscription"]          = np.log1p(df["subscription_times"])

    # Profitability flag
    df["is_profitable"]             = (df["profit_cr"] > 0).astype(int)

    # Leverage band
    df["high_leverage"]             = (df["debt_equity"] > 1).astype(int)

    # Promoter confidence
    df["high_promoter_holding"]     = (df["promoter_holding_pct"] > 60).astype(int)

    # Market context
    df["nifty_normalized"]          = (df["nifty_on_listing"] - df["nifty_on_listing"].mean()) / df["nifty_on_listing"].std()

    # Target variable buckets
    bins   = [-np.inf, -5, 0, 10, 30, 60, np.inf]
    labels = ["Large Loss (< -5%)", "Small Loss (-5–0%)", "Flat (0–10%)",
              "Moderate Gain (10–30%)", "Strong Gain (30–60%)", "Exceptional (> 60%)"]
    df["listing_gain_category"] = pd.cut(df["listing_gain_pct"], bins=bins, labels=labels)

    # OFS vs Fresh vs Mixed
    df["is_ofs"]   = df["issue_type"].str.contains("OFS", na=False).astype(int)
    df["is_fresh"] = df["issue_type"].str.contains("Fresh", na=False).astype(int)

    # Post-listing returns (from base data)
    df["day7_return_pct"]  = ((df["day7_close"]  - df["listing_price"]) / df["listing_price"] * 100).round(2)
    df["day30_return_pct"] = ((df["day30_close"] - df["listing_price"]) / df["listing_price"] * 100).round(2)
    df["day90_return_pct"] = ((df["day90_close"] - df["listing_price"]) / df["listing_price"] * 100).round(2)

    # Decade label
    df["era"] = pd.cut(df["year"], bins=[2012, 2017, 2020, 2025],
                       labels=["2013–2017", "2018–2020", "2021–2024"])

    # ------------------------------------------------------------------
    # Step 4: Optional yfinance enrichment
    # ------------------------------------------------------------------
    if not skip_yfinance:
        logger.info("[4/4] Enriching with yfinance live data (this may take a few minutes)…")

        symbols = df["symbol"].dropna().unique()
        if max_tickers:
            symbols = symbols[:max_tickers]

        yf_records = []
        for sym in symbols:
            row = {"symbol": sym}
            row.update(fetch_yfinance_fundamentals(sym))
            yf_records.append(row)

        yf_df = pd.DataFrame(yf_records)
        df    = df.merge(yf_df, on="symbol", how="left")
        logger.info(f"      yfinance enrichment complete — added {len(yf_df.columns)-1} columns")
    else:
        logger.info("[4/4] Skipping yfinance enrichment (skip_yfinance=True)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv(ENRICHED_PATH, index=False)
    logger.info(f"\n✅ Pipeline complete. Enriched dataset saved → {ENRICHED_PATH}")
    logger.info(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IPO Data Pipeline")
    parser.add_argument("--skip-yfinance", action="store_true",
                        help="Skip yfinance API calls (fast mode)")
    parser.add_argument("--max-tickers", type=int, default=None,
                        help="Limit number of tickers to query")
    args = parser.parse_args()

    df = run_pipeline(max_tickers=args.max_tickers,
                      skip_yfinance=args.skip_yfinance)
    print(df[["company_name", "listing_gain_pct", "listing_gain_category"]].head(20).to_string())
