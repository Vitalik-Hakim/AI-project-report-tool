import yfinance as yf
import pandas as pd
import numpy as np

FIELD_RULES = {
    "receivables": [
        "Receivables",
        "Accounts Receivable",
        "Total Receivables"
    ],

    "revenue": [
        "Total Revenue",
        "Net Sales"
    ],

    "cogs": [
        "Cost Of Revenue",
        "Cost of Revenue"
    ],

    "gross_profit": [
        "Gross Profit"
    ],

    "current_assets": [
        "Total Current Assets",
        "Current Assets"
    ],

    "total_assets": [
        "Total Assets"
    ],

    "ppe": [
        "Net PPE",
        "Property Plant Equipment Net",
        "Property, Plant And Equipment Net"
    ],

    "depreciation": [
        "Reconciled Depreciation",
        "Depreciation And Amortization",
        "Depreciation"
    ],

    "operating_expenses": [
        "Operating Expense",
        "Operating Expenses",
        "Total Operating Expenses"
    ],

    "current_liabilities": [
        "Total Current Liabilities",
        "Current Liabilities"
    ],

    "long_term_debt": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation"
    ],

    "net_income": [
        "Net Income",
        "Net Income Common Stockholders"
    ],

    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities"
    ]
}


# ============================================================
# 2. Find Yahoo Finance field
# ============================================================

def find_field(index_list, possible_names):

    for target in possible_names:

        for field in index_list:

            if str(field).lower() == target.lower():
                return field

    return None


# ============================================================
# 3. Get a financial value
# ============================================================

def get_value(date, field, balance_sheet, financials, cashflow):

    if field is None:
        return np.nan

    if field in balance_sheet.index:
        data = balance_sheet.loc[field]

    elif field in financials.index:
        data = financials.loc[field]

    elif field in cashflow.index:
        data = cashflow.loc[field]

    else:
        return np.nan

    if date not in data.index:
        return np.nan

    value = data.loc[date]

    if pd.isna(value):
        return np.nan

    return float(value)


# ============================================================
# 4. Extract historical data
# ============================================================

def get_beneish_data(ticker):

    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet
    financials = stock.financials
    cashflow = stock.cashflow

    if (
        balance_sheet.empty
        and financials.empty
        and cashflow.empty
    ):
        raise ValueError(
            f"No financial data found for {ticker}"
        )

    # --------------------------------------------------------
    # Get available statement dates
    # --------------------------------------------------------

    dates = set()

    for statement in [
        balance_sheet,
        financials,
        cashflow
    ]:

        if not statement.empty:
            dates.update(statement.columns)

    dates = sorted(
        pd.to_datetime(list(dates)),
        reverse=True
    )

    combined_index = (
        list(balance_sheet.index)
        + list(financials.index)
        + list(cashflow.index)
    )

    rows = []

    # --------------------------------------------------------
    # Extract each year
    # --------------------------------------------------------

    for date in dates:

        row = {
            "ticker": ticker,
            "year": date.year,
            "statement_date": date
        }

        for feature, possible_names in FIELD_RULES.items():

            field = find_field(
                combined_index,
                possible_names
            )

            row[feature] = get_value(
                date,
                field,
                balance_sheet,
                financials,
                cashflow
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ============================================================
# 5. Find two consecutive usable years
# ============================================================

def get_two_years(ticker, target_year):

    df = get_beneish_data(ticker)

    df = df.sort_values(
        "statement_date",
        ascending=False
    ).reset_index(drop=True)

    required = list(FIELD_RULES.keys())

    # Find target year and previous year
    current_rows = df[df["year"] == target_year]
    previous_rows = df[df["year"] == target_year - 1]

    if current_rows.empty or previous_rows.empty:
        return None, None

    current = current_rows.iloc[0]
    previous = previous_rows.iloc[0]

    current_missing = [
        col for col in required
        if pd.isna(current[col])
    ]

    previous_missing = [
        col for col in required
        if pd.isna(previous[col])
    ]

    if current_missing or previous_missing:
        return None, None

    return current, previous


# ============================================================
# 6. Calculate Beneish M-Score
# ============================================================

def calculate_beneish(ticker, target_year):

    current, previous = get_two_years(
        ticker,
        target_year
    )

    if current is None:

        print(
            f"\nNo two consecutive complete "
            f"financial years found for {ticker}."
        )

        return None

    current_year = int(current["year"])
    previous_year = int(previous["year"])

    backtracked = current_year != target_year
    
    if backtracked:

        data_note = (
            f"Requested year {target_year}, but complete "
            f"data was unavailable. Calculation uses "
            f"{current_year} and {previous_year}."
        )

    else:

        data_note = (
            f"Calculation uses requested year "
            f"{current_year} and previous year "
            f"{previous_year}."
        )
    # --------------------------------------------------------
    # Current year values
    # --------------------------------------------------------

    AR_t = current["receivables"]
    REV_t = current["revenue"]
    COGS_t = current["cogs"]
    GP_t = current["gross_profit"]
    CA_t = current["current_assets"]
    TA_t = current["total_assets"]
    PPE_t = current["ppe"]
    DEP_t = current["depreciation"]
    SGA_t = current["operating_expenses"]
    CL_t = current["current_liabilities"]
    LTD_t = current["long_term_debt"]
    NI_t = current["net_income"]
    CFO_t = current["operating_cash_flow"]

    # --------------------------------------------------------
    # Previous year values
    # --------------------------------------------------------

    AR_prev = previous["receivables"]
    REV_prev = previous["revenue"]
    COGS_prev = previous["cogs"]
    GP_prev = previous["gross_profit"]
    CA_prev = previous["current_assets"]
    TA_prev = previous["total_assets"]
    PPE_prev = previous["ppe"]
    DEP_prev = previous["depreciation"]
    SGA_prev = previous["operating_expenses"]
    CL_prev = previous["current_liabilities"]
    LTD_prev = previous["long_term_debt"]

    # ========================================================
    # Beneish indices
    # ========================================================

    # 1. Days Sales in Receivables Index
    DSRI = (
        (AR_t / REV_t)
        /
        (AR_prev / REV_prev)
    )

    # 2. Gross Margin Index
    GMI = (
        ((REV_prev - COGS_prev) / REV_prev)
        /
        ((REV_t - COGS_t) / REV_t)
    )

    # 3. Asset Quality Index
    AQI = (
        (
            1 - (CA_t + PPE_t) / TA_t
        )
        /
        (
            1 - (CA_prev + PPE_prev) / TA_prev
        )
    )

    # 4. Sales Growth Index
    SGI = REV_t / REV_prev

    # 5. Depreciation Index
    DEPI = (
        (DEP_prev / (DEP_prev + PPE_prev))
        /
        (DEP_t / (DEP_t + PPE_t))
    )

    # 6. Sales, General and Administrative Index
    SGAI = (
        (SGA_t / REV_t)
        /
        (SGA_prev / REV_prev)
    )

    # 7. Leverage Index
    LVGI = (
        (
            (CL_t + LTD_t) / TA_t
        )
        /
        (
            (CL_prev + LTD_prev) / TA_prev
        )
    )

    # 8. Total Accruals to Total Assets
    TATA = (
        (NI_t - CFO_t)
        /
        TA_t
    )

    # ========================================================
    # M-Score
    # ========================================================

    M_SCORE = (
        -4.84
        + 0.920 * DSRI
        + 0.528 * GMI
        + 0.404 * AQI
        + 0.892 * SGI
        + 0.115 * DEPI
        - 0.172 * SGAI
        + 4.679 * TATA
        - 0.327 * LVGI
    )

    # ========================================================
    # Interpretation
    # ========================================================

    if M_SCORE > -1.78:

        interpretation = (
            "Possible earnings manipulation"
        )

    else:

        interpretation = (
            "Unlikely to be manipulating earnings"
        )

    df = pd.concat([current, previous])

    # ========================================================
    # Output
    # ========================================================

    return {
        "ticker": ticker,
        "current_year": int(current["year"]),
        "previous_year": int(previous["year"]),
        "DSRI": DSRI,
        "GMI": GMI,
        "AQI": AQI,
        "SGI": SGI,
        "DEPI": DEPI,
        "SGAI": SGAI,
        "LVGI": LVGI,
        "TATA": TATA,
        "M_score": M_SCORE,
        "interpretation": interpretation,
        "backtracked": backtracked,
        "data_note": data_note,
        "data": df.to_dict()

    }