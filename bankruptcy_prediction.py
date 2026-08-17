import joblib
import pandas as pd

from data_extraction import get_company_data, feature_names

# 1. Load trained model and feature columns


from pathlib import Path
_dir = Path(__file__).parent
MODEL_PATH = _dir / "final_rf_calibrated.pkl"
if not MODEL_PATH.exists():
    fallback_model = Path("/Users/vincentchanayire/Downloads/intro-to-ai-project/final_rf_calibrated.pkl")
    if fallback_model.exists():
        MODEL_PATH = fallback_model

FEATURE_COLUMNS_PATH = _dir / "feature_columns.pkl"
if not FEATURE_COLUMNS_PATH.exists():
    fallback_fc = Path("/Users/vincentchanayire/Downloads/intro-to-ai-project/feature_columns.pkl")
    if fallback_fc.exists():
        FEATURE_COLUMNS_PATH = fallback_fc

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
feature_columns = joblib.load(FEATURE_COLUMNS_PATH) if FEATURE_COLUMNS_PATH.exists() else []



# 2. Predict bankruptcy for a company

def predict_bankruptcy(ticker):

    # Get the most recent complete financial year
    company_data = get_company_data(ticker)

    if company_data is None:
        print(f"Could not get complete data for {ticker}.")
        return None


    # Get only the features used by the model
    X = company_data[feature_columns]


    # Get probability of bankruptcy
    probability = model.predict_proba(X)[0][1]


    # Get model's class prediction
    # prediction = model.predict(X)[0]

    return {
        "ticker": ticker,
        "year": int(company_data["year"].iloc[0]),
        "bankruptcy_probability": probability,
        "data": company_data.rename(columns=feature_names)
        # "prediction": int(prediction)
    }