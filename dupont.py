"""
=============================================================================
CS 254: Introduction to Artificial Intelligence — Final Project
DuPont 3-Step Return on Equity (ROE) Decomposition Engine
=============================================================================

Academic Citations & Theoretical Grounding:
- Donaldson, G. (1961). Corporate Debt Capacity. Harvard Business School.
- Soliman, M. T. (2008). The Use of DuPont Analysis by Market Participants.
  The Accounting Review, 83(3), 823-853.

Mathematical Formulation:
  ROE = Net Profit Margin * Asset Turnover * Financial Leverage Multiplier
  Where:
    - Net Profit Margin = Net Income / Revenue (Operational Profitability)
    - Asset Turnover = Revenue / Total Assets (Capital Efficiency)
    - Equity Multiplier = Total Assets / Total Stockholders' Equity (Leverage)

Core Diligence Role:
  Distinguishes between genuine operational profitability (margin & efficiency)
  versus financial engineering driven by high debt leverage.
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# Fields required for DuPont Analysis
# ============================================================

DUPONT_FIELDS = {
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders"
    ],
    "total_revenue": [
        "Total Revenue",
        "Net Sales"
    ],
    "total_assets": [
        "Total Assets"
    ],
    "total_equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity"
    ]
}


def find_field(index_list, possible_names):
    for target in possible_names:
        for field in index_list:
            if str(field).lower() == target.lower():
                return field
    return None


def get_dupont_data(ticker, target_year):
    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet
    financials = stock.financials

    if balance_sheet.empty and financials.empty:
        print(f"No financial data found for {ticker}.")
        return None

    dates = set()
    if not balance_sheet.empty:
        dates.update(balance_sheet.columns)
    if not financials.empty:
        dates.update(financials.columns)

    dates = sorted(pd.to_datetime(list(dates)), reverse=True)
    combined_index = list(balance_sheet.index) + list(financials.index)

    rows = []
    for date in dates:
        if date.year > target_year:
            continue

        row = {
            "ticker": ticker,
            "year": date.year,
            "statement_date": date
        }

        for name, possible_names in DUPONT_FIELDS.items():
            field = find_field(combined_index, possible_names)
            if field is None:
                row[name] = np.nan
                continue

            if field in balance_sheet.index:
                data = balance_sheet.loc[field]
            elif field in financials.index:
                data = financials.loc[field]
            else:
                row[name] = np.nan
                continue

            if date in data.index:
                val = data.loc[date]
                row[name] = float(val) if not pd.isna(val) else np.nan
            else:
                row[name] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Filter rows with complete DuPont fields
    required = list(DUPONT_FIELDS.keys())
    usable = df.dropna(subset=required).sort_values("year", ascending=False)

    if usable.empty:
        return None

    current = usable.iloc[0]
    previous = usable.iloc[1] if len(usable) > 1 else None

    return current, previous


def calculate_dupont(ticker, target_year):
    result = get_dupont_data(ticker, target_year)
    if result is None:
        return None

    current, previous = result
    current_year = int(current["year"])
    backtracked = current_year != target_year

    if backtracked:
        data_note = (
            f"Requested year {target_year}, but complete data was unavailable. "
            f"DuPont calculation uses {current_year}."
        )
    else:
        data_note = f"DuPont calculation uses requested year {current_year}."

    net_income = current["net_income"]
    revenue = current["total_revenue"]
    total_assets = current["total_assets"]
    total_equity = current["total_equity"]

    if revenue == 0 or total_assets == 0 or total_equity == 0:
        return None

    # 3-Step DuPont Breakdown
    profit_margin = net_income / revenue
    asset_turnover = revenue / total_assets
    equity_multiplier = total_assets / total_equity
    roe = profit_margin * asset_turnover * equity_multiplier

    # Compare with previous year if available
    trend = {}
    driver = "Balanced"
    if previous is not None and not pd.isna(previous["total_revenue"]):
        prev_pm = previous["net_income"] / previous["total_revenue"] if previous["total_revenue"] != 0 else np.nan
        prev_at = previous["total_revenue"] / previous["total_assets"] if previous["total_assets"] != 0 else np.nan
        prev_em = previous["total_assets"] / previous["total_equity"] if previous["total_equity"] != 0 else np.nan
        prev_roe = prev_pm * prev_at * prev_em if not (np.isnan(prev_pm) or np.isnan(prev_at) or np.isnan(prev_em)) else np.nan

        trend = {
            "previous_year": int(previous["year"]),
            "previous_profit_margin": prev_pm,
            "previous_asset_turnover": prev_at,
            "previous_equity_multiplier": prev_em,
            "previous_roe": prev_roe,
            "roe_change": roe - prev_roe if not np.isnan(prev_roe) else None
        }

        # Detect primary return driver
        if not np.isnan(prev_em) and prev_em > 0:
            em_pct_change = (equity_multiplier - prev_em) / prev_em
            pm_pct_change = (profit_margin - prev_pm) / abs(prev_pm) if prev_pm != 0 else 0
            at_pct_change = (asset_turnover - prev_at) / prev_at if prev_at != 0 else 0

            if em_pct_change > 0.20 and em_pct_change > max(pm_pct_change, at_pct_change):
                driver = "Leverage-Driven"
            elif pm_pct_change > 0.10 or at_pct_change > 0.10:
                driver = "Operational-Driven"
            else:
                driver = "Stable Operating Base"

    # Evaluation / Classification
    if roe >= 0.20:
        classification = "High Quality / Strong Return"
    elif roe >= 0.10:
        classification = "Moderate Return"
    elif roe > 0:
        classification = "Low Return"
    else:
        classification = "Negative Return"

    return {
        "ticker": ticker,
        "requested_year": target_year,
        "year": current_year,
        "backtracked": backtracked,
        "data_note": data_note,
        "profit_margin": profit_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": equity_multiplier,
        "roe": roe,
        "primary_driver": driver,
        "classification": classification,
        "trend": trend
    }
