

import pandas as pd
import numpy as np
import yfinance as yf

ROIC_WACC_FIELDS = {
    "ebit": [
        "EBIT",
        "Operating Income",
        "Total Operating Income As Reported"
    ],
    "tax_provision": [
        "Tax Provision"
    ],
    "interest_expense": [
        "Interest Expense Non Operating",
        "Interest Expense"
    ],
    "total_assets": [
        "Total Assets"
    ],
    "cash": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash"
    ],
    "current_liabilities": [
        "Total Current Liabilities",
        "Current Liabilities"
    ],
    "long_term_debt": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation"
    ],
    "total_equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest"
    ]
}


def find_field(index_list, possible_names):
    for target in possible_names:
        for field in index_list:
            if str(field).lower() == target.lower():
                return field
    return None


def get_roic_wacc_data(ticker, target_year):
    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet
    financials = stock.financials

    if balance_sheet.empty and financials.empty:
        print(f"No balance sheet/financials found for {ticker}.")
        return None, None

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

        for name, possible_names in ROIC_WACC_FIELDS.items():
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
        return None, None

    # Filter rows where at least total_assets and ebit exist
    usable = df.dropna(subset=["total_assets", "ebit"]).sort_values("year", ascending=False)
    if usable.empty:
        return None, None

    current = usable.iloc[0]
    return current, stock


def calculate_roic_wacc(
    ticker,
    target_year,
    risk_free_rate=0.0425,       # 4.25% US 10-Yr Benchmark
    market_risk_premium=0.055,   # 5.50% Market Risk Premium
    beta_override=None
):
    current, stock = get_roic_wacc_data(ticker, target_year)
    if current is None:
        return None

    current_year = int(current["year"])
    backtracked = current_year != target_year

    if backtracked:
        data_note = (
            f"Requested year {target_year}, but complete data was unavailable. "
            f"ROIC/WACC calculation uses {current_year}."
        )
    else:
        data_note = f"ROIC/WACC calculation uses requested year {current_year}."

    ebit = current["ebit"]
    tax_provision = current["tax_provision"] if not pd.isna(current["tax_provision"]) else None
    interest_expense = current["interest_expense"] if not pd.isna(current["interest_expense"]) else 0.0
    total_assets = current["total_assets"]
    cash = current["cash"] if not pd.isna(current["cash"]) else 0.0
    current_liabilities = current["current_liabilities"] if not pd.isna(current["current_liabilities"]) else 0.0
    long_term_debt = current["long_term_debt"] if not pd.isna(current["long_term_debt"]) else 0.0
    total_equity = current["total_equity"] if not pd.isna(current["total_equity"]) else 0.0

    # 1. Effective Tax Rate & NOPAT
    ebt = ebit - interest_expense
    if ebt > 0 and tax_provision is not None and tax_provision > 0:
        effective_tax_rate = tax_provision / ebt
        # Clamp tax rate between 0% and 35%
        effective_tax_rate = max(0.0, min(0.35, effective_tax_rate))
    else:
        effective_tax_rate = 0.21  # Statutory benchmark default

    nopat = ebit * (1.0 - effective_tax_rate)

    # Standard formula: Total Assets - Excess Cash - Non-Interest-Bearing Current Liab
    invested_capital = total_assets - cash - current_liabilities
    if invested_capital <= 0:
        # Fallback to Equity + Long Term Debt - Cash
        invested_capital = max(1.0, (total_equity + long_term_debt - cash))

    roic = nopat / invested_capital if invested_capital > 0 else 0.0

    # Get Beta from stock info or fallback to 1.0
    beta = 1.0
    market_cap = None
    if beta_override is not None:
        beta = beta_override
    else:
        try:
            info = stock.info
            beta = info.get("beta", 1.0)
            if beta is None or pd.isna(beta):
                beta = 1.0
            market_cap = info.get("marketCap", None)
        except Exception:
            beta = 1.0

    if market_cap is None or market_cap <= 0:
        market_cap = max(total_equity, total_assets)

    # Capital weights
    total_firm_value = market_cap + long_term_debt
    if total_firm_value > 0:
        we = market_cap / total_firm_value
        wd = long_term_debt / total_firm_value
    else:
        we, wd = 0.8, 0.2

    # Cost of Equity via CAPM: Ke = Rf + Beta * MRP
    cost_of_equity = risk_free_rate + (beta * market_risk_premium)

    # Cost of Debt: Kd = Interest Expense / Long Term Debt
    if long_term_debt > 0 and interest_expense > 0:
        pre_tax_cost_of_debt = interest_expense / long_term_debt
        # Clamp pre-tax cost of debt to realistic bounds (2% - 12%)
        pre_tax_cost_of_debt = max(0.02, min(0.12, pre_tax_cost_of_debt))
    else:
        pre_tax_cost_of_debt = risk_free_rate + 0.015  # Benchmark BBB spread

    after_tax_cost_of_debt = pre_tax_cost_of_debt * (1.0 - effective_tax_rate)

    # WACC = We * Ke + Wd * Kd * (1 - t)
    wacc = (we * cost_of_equity) + (wd * after_tax_cost_of_debt)

    # 4. Economic Value Added (EVA) Spread
    economic_spread = roic - wacc

    if economic_spread >= 0.05:
        value_creation = "Strong Value Creation"
    elif economic_spread > 0.0:
        value_creation = "Creating Value"
    elif economic_spread >= -0.02:
        value_creation = "Neutral / Near Cost of Capital"
    else:
        value_creation = "Destroying Value"

    return {
        "ticker": ticker,
        "requested_year": target_year,
        "year": current_year,
        "backtracked": backtracked,
        "data_note": data_note,
        "nopat": nopat,
        "invested_capital": invested_capital,
        "roic": roic,
        "beta": beta,
        "risk_free_rate": risk_free_rate,
        "market_risk_premium": market_risk_premium,
        "cost_of_equity": cost_of_equity,
        "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "equity_weight": we,
        "debt_weight": wd,
        "wacc": wacc,
        "economic_spread": economic_spread,
        "value_creation": value_creation
    }
