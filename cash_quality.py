"""
=============================================================================
CS 254: Introduction to Artificial Intelligence — Final Project
Cash Flow Quality & Conversion Analysis Engine
=============================================================================

Academic Citations & Theoretical Grounding:
- Sloan, R. G. (1996). Do Stock Prices Fully Reflect Information in Accruals
  and Cash Flows about Future Earnings? The Accounting Review, 71(3), 289-315.
- Dechow, P. M., & Dichev, I. D. (2002). The Quality of Accruals and Earnings:
  The Role of Accrual Estimation Errors. The Accounting Review, 77(s-1), 35-59.

Mathematical Formulation:
  - Free Cash Flow (FCF) = Operating Cash Flow (OCF) - Capital Expenditures (CapEx)
  - Cash Conversion Ratio (CCR) = Operating Cash Flow / Net Income
  - FCF Conversion Ratio = Free Cash Flow / Net Income

Core Diligence Role:
  Assesses whether accounting net income is supported by genuine cash inflows,
  identifies aggressive working capital accruals, and tracks multi-year FCF trends.
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf

# ============================================================
# Fields required for Cash Quality Analysis
# ============================================================

CASH_QUALITY_FIELDS = {
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders"
    ],
    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities"
    ],
    "capital_expenditure": [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE",
        "Purchase Of Property Plant And Equipment"
    ]
}


def find_field(index_list, possible_names):
    for target in possible_names:
        for field in index_list:
            if str(field).lower() == target.lower():
                return field
    return None


def get_cash_quality_data(ticker, target_year):
    stock = yf.Ticker(ticker)

    financials = stock.financials
    cashflow = stock.cashflow

    if financials.empty and cashflow.empty:
        print(f"No financial statement data found for {ticker}.")
        return None

    dates = set()
    if not financials.empty:
        dates.update(financials.columns)
    if not cashflow.empty:
        dates.update(cashflow.columns)

    dates = sorted(pd.to_datetime(list(dates)), reverse=True)
    combined_index = list(financials.index) + list(cashflow.index)

    rows = []
    for date in dates:
        if date.year > target_year:
            continue

        row = {
            "ticker": ticker,
            "year": date.year,
            "statement_date": date
        }

        for name, possible_names in CASH_QUALITY_FIELDS.items():
            field = find_field(combined_index, possible_names)
            if field is None:
                row[name] = np.nan
                continue

            if field in financials.index:
                data = financials.loc[field]
            elif field in cashflow.index:
                data = cashflow.loc[field]
            else:
                row[name] = np.nan
                continue

            if date in data.index:
                val = data.loc[date]
                if pd.isna(val):
                    row[name] = np.nan
                else:
                    val = float(val)
                    if name == "capital_expenditure":
                        val = abs(val)
                    row[name] = val
            else:
                row[name] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    # Filter rows with complete data
    usable = df.dropna(subset=["net_income", "operating_cash_flow"]).sort_values("year", ascending=False)
    if usable.empty:
        return None

    current = usable.iloc[0]
    previous = usable.iloc[1] if len(usable) > 1 else None

    return current, previous


def calculate_cash_quality(ticker, target_year):
    result = get_cash_quality_data(ticker, target_year)
    if result is None:
        return None

    current, previous = result
    current_year = int(current["year"])
    backtracked = current_year != target_year

    if backtracked:
        data_note = (
            f"Requested year {target_year}, but complete cash flow data was unavailable. "
            f"Calculation uses {current_year}."
        )
    else:
        data_note = f"Cash quality calculation uses requested year {current_year}."

    net_income = current["net_income"]
    operating_cash_flow = current["operating_cash_flow"]
    capital_expenditure = current["capital_expenditure"] if not pd.isna(current["capital_expenditure"]) else 0.0

    free_cash_flow = operating_cash_flow - capital_expenditure

    # 1. Cash Conversion (OCF / Net Income)
    if net_income != 0:
        cash_conversion_ratio = operating_cash_flow / net_income
        fcf_conversion_ratio = free_cash_flow / net_income
    else:
        cash_conversion_ratio = None
        fcf_conversion_ratio = None

    # 2. Previous Year Trend
    fcf_trend = "Stable"
    prev_fcf_conversion = None
    if previous is not None and not pd.isna(previous["net_income"]) and previous["net_income"] != 0:
        prev_ocf = previous["operating_cash_flow"]
        prev_capex = previous["capital_expenditure"] if not pd.isna(previous["capital_expenditure"]) else 0.0
        prev_fcf = prev_ocf - prev_capex
        prev_fcf_conversion = prev_fcf / previous["net_income"]

        if fcf_conversion_ratio is not None:
            if fcf_conversion_ratio > prev_fcf_conversion + 0.05:
                fcf_trend = "Improving"
            elif fcf_conversion_ratio < prev_fcf_conversion - 0.05:
                fcf_trend = "Declining"

    # 3. Conversion Level
    if fcf_conversion_ratio is None:
        conversion_level = "Unknown"
    elif fcf_conversion_ratio >= 0.80:
        conversion_level = "Strong"
    elif fcf_conversion_ratio >= 0.50:
        conversion_level = "Moderate"
    else:
        conversion_level = "Weak"

    # 4. Overall Cash Quality Rating
    if conversion_level == "Strong" and fcf_trend != "Declining":
        overall_cash_quality = "Strong"
    elif conversion_level == "Weak" or (fcf_conversion_ratio is not None and fcf_conversion_ratio < 0):
        overall_cash_quality = "Weak"
    elif conversion_level == "Moderate" or fcf_trend == "Declining":
        overall_cash_quality = "Watch"
    else:
        overall_cash_quality = "Moderate"

    return {
        "ticker": ticker,
        "requested_year": target_year,
        "year": current_year,
        "backtracked": backtracked,
        "data_note": data_note,
        "operating_cash_flow": operating_cash_flow,
        "capital_expenditure": capital_expenditure,
        "free_cash_flow": free_cash_flow,
        "cash_conversion_ratio": cash_conversion_ratio,
        "fcf_conversion_ratio": fcf_conversion_ratio,
        "fcf_trend": fcf_trend,
        "conversion_level": conversion_level,
        "overall_cash_quality": overall_cash_quality
    }
