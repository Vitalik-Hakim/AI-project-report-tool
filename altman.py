import pandas as pd

from data_extraction import get_company_data, feature_names


def calculate_altman_z(ticker, target_year):

    data = get_company_data(ticker)

    if data is None or data.empty:
        print(
            f"No complete Altman data available "
            f"for {ticker}."
        )
        return None



    usable_data = data[
        data["year"] <= target_year
    ].copy()

    # Keep only rows where all 18 Altman features are available
    usable_data = usable_data.dropna(
        subset=[
            "X1", "X2", "X3", "X4",
            "X5", "X6", "X7", "X8",
            "X9", "X10", "X11", "X12",
            "X13", "X14", "X15", "X16",
            "X17", "X18"
        ]
    )

    usable_data = usable_data.sort_values(
        "year",
        ascending=False
    )

    if usable_data.empty:
        print(
            f"No Altman data available for {ticker} "
            f"for {target_year} or earlier."
        )
        return None

    row = usable_data.iloc[0]

    actual_year = int(row["year"])


    backtracked = actual_year != target_year

    if backtracked:

        data_note = (
            f"Requested year {target_year}, but complete "
            f"data was unavailable. Calculation uses "
            f"{actual_year}."
        )

    else:

        data_note = (
            f"Complete data was available for {target_year}."
        )

    # --------------------------------------------------------
    # Altman Z-Score components
    # --------------------------------------------------------

    x1 = (
        row["X1"]
        - row["X14"]
    ) / row["X10"]

    x2 = (
        row["X15"]
        / row["X10"]
    )

    x3 = (
        row["X12"]
        / row["X10"]
    )

    x4 = (
        row["X8"]
        / row["X17"]
    )

    x5 = (
        row["X16"]
        / row["X10"]
    )


    z_score = (
        1.2 * x1
        + 1.4 * x2
        + 3.3 * x3
        + 0.6 * x4
        + 1.0 * x5
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if z_score > 2.99:
        classification = "Safe Zone"

    elif z_score >= 1.81:
        classification = "Grey Zone"

    else:
        classification = "Distress Zone"

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "ticker": ticker,
        "requested_year": target_year,
        "year": actual_year,
        "backtracked": backtracked,
        "data_note": data_note,
        "altman_z_score": z_score,
        "altman_classification": classification,
        "data": data.rename(columns=feature_names).to_dict()
    }