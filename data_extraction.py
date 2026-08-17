import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# 1. Features required by the trained model
# ============================================================

feature_names = {
    "X1": "Current Assets",
    "X2": "Cost of Goods Sold",
    "X3": "Depreciation and Amortization",
    "X4": "EBITDA",
    "X5": "Inventory",
    "X6": "Net Income",
    "X7": "Total Receivables",
    "X8": "Market Value",
    "X9": "Net Sales",
    "X10": "Total Assets",
    "X11": "Total Long-term Debt",
    "X12": "EBIT",
    "X13": "Gross Profit",
    "X14": "Total Current Liabilities",
    "X15": "Retained Earnings",
    "X16": "Total Revenue",
    "X17": "Total Liabilities",
    "X18": "Total Operating Expenses",
}

FIELD_RULES = {
    "X1": {
        "exact": ["Total Current Assets", "Current Assets"]
    },

    "X2": {
        "exact": ["Cost Of Revenue", "Cost of Revenue"]
    },

    "X3": {
        "exact": [
            "Reconciled Depreciation",
            "Depreciation And Amortization",
            "Depreciation"
        ]
    },

    "X4": {
        "exact": ["EBITDA"]
    },

    "X5": {
        "exact": ["Inventory"]
    },

    "X6": {
        "exact": [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    },

    "X7": {
        "exact": [
            "Receivables",
            "Accounts Receivable",
            "Total Receivables"
        ]
    },

    "X8": {
        "special": "marketCap"
    },

    "X9": {
        "exact": [
            "Net Sales",
            "Total Revenue"
        ]
    },

    "X10": {
        "exact": ["Total Assets"]
    },

    "X11": {
        "exact": [
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    },

    "X12": {
        "exact": [
            "EBIT",
            "Operating Income",
            "Total Operating Income As Reported"
        ]
    },

    "X13": {
        "exact": ["Gross Profit"]
    },

    "X14": {
        "exact": [
            "Total Current Liabilities",
            "Current Liabilities"
        ]
    },

    "X15": {
        "exact": ["Retained Earnings"]
    },

    "X16": {
        "exact": ["Total Revenue"]
    },

    "X17": {
        "exact": [
            "Total Liabilities Net Minority Interest",
            "Total Liabilities"
        ]
    },

    "X18": {
        "exact": [
            "Operating Expense",
            "Operating Expenses",
            "Total Operating Expenses"
        ]
    }
}


MODEL_FEATURES = list(FIELD_RULES.keys())


# ============================================================
# 2. Find Yahoo Finance field
# ============================================================

def find_field(index_list, rules):

    for target in rules.get("exact", []):

        for field in index_list:

            if str(field).lower() == target.lower():
                return field

    return None


# ============================================================
# 3. Historical market cap
# ============================================================

def get_historical_market_cap(stock, statement_date):

    try:

        date = pd.Timestamp(statement_date)

        # ----------------------------------------------------
        # Historical shares outstanding
        # ----------------------------------------------------

        shares = stock.get_shares_full(
            start=date - pd.Timedelta(days=30),
            end=date + pd.Timedelta(days=1)
        )

        if shares is None or shares.empty:
            return np.nan

        # Remove timezone
        shares.index = pd.to_datetime(
            shares.index
        ).tz_localize(None)

        # Only use shares known on/before statement date
        shares_before_date = shares[
            shares.index <= date
        ]

        if shares_before_date.empty:
            return np.nan

        shares_outstanding = float(
            shares_before_date.iloc[-1]
        )

        # ----------------------------------------------------
        # Historical stock price
        # ----------------------------------------------------

        hist = stock.history(
            start=date - pd.Timedelta(days=5),
            end=date + pd.Timedelta(days=5),
            auto_adjust=False
        )

        if hist.empty:
            return np.nan

        # Remove timezone
        hist.index = pd.to_datetime(
            hist.index
        ).tz_localize(None)

        # Find closest trading day
        closest_date = min(
            hist.index,
            key=lambda x: abs(x - date)
        )

        price = float(
            hist.loc[closest_date, "Close"]
        )

        return price * shares_outstanding

    except Exception as e:

        print(
            f"Market cap error for "
            f"{stock.ticker} on {statement_date}: {e}"
        )

        return np.nan


# ============================================================
# 4. Get historical financial data
# ============================================================

def get_company_data(ticker):

    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet
    financials = stock.financials

    if balance_sheet.empty and financials.empty:
        raise ValueError(
            f"No financial data found for {ticker}"
        )

    # --------------------------------------------------------
    # Get all available financial statement dates
    # --------------------------------------------------------

    dates = set()

    if not balance_sheet.empty:
        dates.update(balance_sheet.columns)

    if not financials.empty:
        dates.update(financials.columns)

    dates = sorted(
        pd.to_datetime(list(dates)),
        reverse=True
    )

    combined_index = (
        list(balance_sheet.index)
        + list(financials.index)
    )

    rows = []

    # --------------------------------------------------------
    # Extract every available year
    # --------------------------------------------------------

    for date in dates:

        row = {
            "ticker": ticker,
            "year": date.year,
            "statement_date": date
        }

        # ----------------------------------------------------
        # Extract X1-X18
        # ----------------------------------------------------

        for feature, rules in FIELD_RULES.items():

            # X8 is calculated separately
            if rules.get("special") == "marketCap":
                continue

            field = find_field(
                combined_index,
                rules
            )

            if field is None:
                row[feature] = np.nan
                continue

            # Determine which statement contains field
            if field in balance_sheet.index:
                data = balance_sheet.loc[field]

            elif field in financials.index:
                data = financials.loc[field]

            else:
                row[feature] = np.nan
                continue

            # Get value for this exact date
            if date in data.index:

                value = data.loc[date]

                if pd.isna(value):
                    row[feature] = np.nan

                else:
                    row[feature] = float(value)

            else:
                row[feature] = np.nan

        # ----------------------------------------------------
        # X8 - Historical market value
        # ----------------------------------------------------

        row["X8"] = get_historical_market_cap(
            stock,
            date
        )

        rows.append(row)

    # --------------------------------------------------------
    # Create historical DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # Newest → oldest
    df = df.sort_values(
        "statement_date",
        ascending=False
    ).reset_index(drop=True)

    return df


def find_analysis_year(data):

    if data is None or data.empty:
        return None

    for _, row in data.iterrows():

        missing_features = [
            feature
            for feature in MODEL_FEATURES
            if pd.isna(row[feature])
        ]

        if not missing_features:
            return int(row["year"])

    return None

# aapl_data = get_company_data("AAPL")


# candidates = [
#     "MSFT",
#     "AAPL",
#     "JNJ",
#     "PG",
#     "KO",
#     "PEP",
#     "XOM",
#     "CVX",
#     "JPM",
#     "V",
#     "WMT",
#     "MCD",
#     "HD",
#     "IBM",
#     "CSCO",
# ]

# for candidate in candidates:
#     data = get_company_data(candidate)
#     if data is not None:

#         print("\nFINAL MODEL INPUT")
#         print(
#             data.to_string(index=False)
#         )

#     else:

#         print(
#             "No usable data found."
#         )