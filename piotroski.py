import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# Fields required for Piotroski F-Score
# ============================================================

PIOTROSKI_FIELDS = {
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders"
    ],

    "operating_cash_flow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities"
    ],

    "total_assets": [
        "Total Assets"
    ],

    "current_assets": [
        "Total Current Assets",
        "Current Assets"
    ],

    "current_liabilities": [
        "Total Current Liabilities",
        "Current Liabilities"
    ],

    "long_term_debt": [
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation"
    ],

    "gross_profit": [
        "Gross Profit"
    ],

    "total_revenue": [
        "Total Revenue"
    ]
}


def find_field(index_list, possible_names):

    for target in possible_names:

        for field in index_list:

            if str(field).lower() == target.lower():
                return field

    return None


# ============================================================
# Get two usable years at or before target year
# ============================================================

def get_piotroski_data(ticker, target_year):

    stock = yf.Ticker(ticker)

    balance_sheet = stock.balance_sheet
    financials = stock.financials
    cashflow = stock.cashflow

    if (
        balance_sheet.empty
        and financials.empty
        and cashflow.empty
    ):
        print(f"No financial data found for {ticker}.")
        return None

    # --------------------------------------------------------
    # Financial statement dates
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

    # --------------------------------------------------------
    # Available fields
    # --------------------------------------------------------

    combined_index = (
        list(balance_sheet.index)
        + list(financials.index)
        + list(cashflow.index)
    )

    rows = []

    # --------------------------------------------------------
    # Extract all available years
    # --------------------------------------------------------

    for date in dates:

        # Don't use a year later than requested
        if date.year > target_year:
            continue

        row = {
            "ticker": ticker,
            "year": date.year,
            "statement_date": date
        }

        # --------------------------------------------
        # Extract required fields
        # --------------------------------------------

        for name, possible_names in PIOTROSKI_FIELDS.items():

            field = find_field(
                combined_index,
                possible_names
            )

            if field is None:
                row[name] = np.nan
                continue

            if field in balance_sheet.index:
                data = balance_sheet.loc[field]

            elif field in financials.index:
                data = financials.loc[field]

            elif field in cashflow.index:
                data = cashflow.loc[field]

            else:
                row[name] = np.nan
                continue

            if date in data.index:

                value = data.loc[date]

                if pd.isna(value):
                    row[name] = np.nan

                else:
                    row[name] = float(value)

            else:
                row[name] = np.nan

        # --------------------------------------------
        # Shares outstanding
        # --------------------------------------------

        try:

            shares = stock.get_shares_full(
                start=date - pd.Timedelta(days=30),
                end=date + pd.Timedelta(days=1)
            )

            if shares is not None and not shares.empty:

                shares.index = pd.to_datetime(
                    shares.index
                ).tz_localize(None)

                shares_before = shares[
                    shares.index <= date
                ]

                if not shares_before.empty:

                    row["shares_outstanding"] = float(
                        shares_before.iloc[-1]
                    )

                else:

                    row["shares_outstanding"] = np.nan

            else:

                row["shares_outstanding"] = np.nan

        except Exception:

            row["shares_outstanding"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return None

    # --------------------------------------------------------
    # Search backward for two consecutive complete years
    # --------------------------------------------------------

    required = list(PIOTROSKI_FIELDS.keys()) + [
        "shares_outstanding"
    ]

    for current_year in range(
        target_year,
        int(df["year"].min()) + 1,
        -1
    ):

        previous_year = current_year - 1

        current_rows = df[
            df["year"] == current_year
        ]

        previous_rows = df[
            df["year"] == previous_year
        ]

        if current_rows.empty or previous_rows.empty:
            continue

        current = current_rows.iloc[0]
        previous = previous_rows.iloc[0]

        current_missing = [
            field
            for field in required
            if pd.isna(current[field])
        ]

        previous_missing = [
            field
            for field in required
            if pd.isna(previous[field])
        ]

        if not current_missing and not previous_missing:

            return pd.DataFrame(
                [current, previous]
            ).reset_index(drop=True)

    # --------------------------------------------------------
    # No usable pair found
    # --------------------------------------------------------

    print(
        f"Could not find two consecutive complete "
        f"years for {ticker} at or before {target_year}."
    )

    return None

def calculate_piotroski(ticker, target_year):

    data = get_piotroski_data(
        ticker,
        target_year
    )

    score = 0
    signals = {}

    if data is None:
        return None

    current = data.iloc[0]
    previous = data.iloc[1]

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

    # ========================================================
    # 1. Positive Net Income
    # ========================================================

    signals["positive_net_income"] = int(
        current["net_income"] > 0
    )

    # ========================================================
    # 2. Positive Operating Cash Flow
    # ========================================================

    signals["positive_operating_cash_flow"] = int(
        current["operating_cash_flow"] > 0
    )

    # ========================================================
    # 3. Improving ROA
    # ========================================================

    roa_current = (
        current["net_income"]
        / current["total_assets"]
    )

    roa_previous = (
        previous["net_income"]
        / previous["total_assets"]
    )

    signals["improving_roa"] = int(
        roa_current > roa_previous
    )

    # ========================================================
    # 4. Operating Cash Flow > Net Income
    # ========================================================

    signals["cash_flow_greater_than_net_income"] = int(
        current["operating_cash_flow"]
        > current["net_income"]
    )

    # ========================================================
    # 5. Lower Leverage
    # ========================================================

    leverage_current = (
        current["long_term_debt"]
        / current["total_assets"]
    )

    leverage_previous = (
        previous["long_term_debt"]
        / previous["total_assets"]
    )

    signals["lower_leverage"] = int(
        leverage_current < leverage_previous
    )

    # ========================================================
    # 6. Improving Current Ratio
    # ========================================================

    current_ratio_current = (
        current["current_assets"]
        / current["current_liabilities"]
    )

    current_ratio_previous = (
        previous["current_assets"]
        / previous["current_liabilities"]
    )

    signals["improving_current_ratio"] = int(
        current_ratio_current > current_ratio_previous
    )

    # ========================================================
    # 7. No New Shares Issued
    # ========================================================

    signals["no_new_shares"] = int(
        current["shares_outstanding"]
        <= previous["shares_outstanding"]
    )

    # ========================================================
    # 8. Improving Gross Margin
    # ========================================================

    gross_margin_current = (
        current["gross_profit"]
        / current["total_revenue"]
    )

    gross_margin_previous = (
        previous["gross_profit"]
        / previous["total_revenue"]
    )

    signals["improving_gross_margin"] = int(
        gross_margin_current > gross_margin_previous
    )

    # ========================================================
    # 9. Improving Asset Turnover
    # ========================================================

    asset_turnover_current = (
        current["total_revenue"]
        / current["total_assets"]
    )

    asset_turnover_previous = (
        previous["total_revenue"]
        / previous["total_assets"]
    )

    signals["improving_asset_turnover"] = int(
        asset_turnover_current > asset_turnover_previous
    )

    # ========================================================
    # Total Score
    # ========================================================

    score = sum(signals.values())

    if score >= 7:
        classification = "Strong"

    elif score >= 4:
        classification = "Moderate"

    else:
        classification = "Weak"


    return {
        "ticker": ticker,
        "requested_year": target_year,
        "current_year": current_year,
        "previous_year": previous_year,
        "backtracked": backtracked,
        "data_note": data_note,
        "piotroski_score": score,
        "classification": classification,
        "signals": signals,
        "data": data.to_dict()
    }