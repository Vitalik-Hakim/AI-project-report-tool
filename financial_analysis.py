
import os
import datetime
from dotenv import load_dotenv
load_dotenv()
try:
    from mistralai.client import Mistral
except ImportError:
    try:
        from mistralai import Mistral
    except ImportError:
        Mistral = None
from bankruptcy_prediction import predict_bankruptcy
from altman import calculate_altman_z
from piotroski import calculate_piotroski
from beneish import calculate_beneish
from dupont import calculate_dupont
from cash_quality import calculate_cash_quality
from roic_wacc import calculate_roic_wacc

API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if (Mistral and API_KEY) else None


def analyze_company(ticker):
    """
    Runs the complete 5-layer financial analysis pipeline for a given ticker:
    1. Supervised ML Distress Prediction
    2. Altman Z-Score (Distress Zone)
    3. Piotroski F-Score (Fundamental Quality)
    4. Beneish M-Score (Earnings Manipulation Screening)
    5. DuPont Analysis (Operational vs. Leverage Drivers)
    6. Cash Quality (FCF & Cash Conversion)
    7. ROIC vs. WACC (Value Creation Spread)
    """

    # 1. Supervised Bankruptcy / Distress Prediction
    bankruptcy = predict_bankruptcy(ticker)
    analysis_year = bankruptcy["year"] if (bankruptcy and "year" in bankruptcy) else datetime.date.today().year - 1

    # 2. Altman Z-Score
    altman = calculate_altman_z(ticker, analysis_year)

    # 3. Piotroski F-Score
    piotroski = calculate_piotroski(ticker, analysis_year)

    # 4. Beneish M-Score
    beneish = calculate_beneish(ticker, analysis_year)

    # 5. DuPont Analysis (ROE Breakdown)
    dupont = calculate_dupont(ticker, analysis_year)

    # 6. Cash Quality Engine
    cash_quality = calculate_cash_quality(ticker, analysis_year)

    # 7. ROIC vs. WACC (Economic Spread)
    roic_wacc = calculate_roic_wacc(ticker, analysis_year)

    return {
        "ticker": ticker,
        "analysis_year": analysis_year,
        "bankruptcy": bankruptcy,
        "altman": altman,
        "piotroski": piotroski,
        "beneish": beneish,
        "dupont": dupont,
        "cash_quality": cash_quality,
        "roic_wacc": roic_wacc
    }


def generate_llm_report(result):
    """
    Constrained LLM narrative generator turning deterministic results into a PE/M&A screening report.
    Adheres strictly to hallucination and recalculation guardrails.
    """
    if not client:
        return ["Mistral API key not configured. Generated report unavailable.", None]

    SYSTEM_PROMPT = """
    You are a senior financial analysis assistant producing a preliminary standing report for a PE/M&A target screening workflow.

    Your task is to interpret structured financial analysis produced by a deterministic Python analysis pipeline and explain the results clearly and concisely.

    IMPORTANT NON-NEGOTIABLE RULES:
    1. Do NOT recalculate any financial score, ratio, probability, spread, or metric. Treat all supplied calculated values as authoritative facts.
    2. Do NOT invent financial information. If information is missing or None, explicitly state that it is unavailable.
    3. Use ONLY the information provided in the prompt. Do not introduce outside facts or assumptions (e.g. do not guess about unmentioned market conditions, mergers, or buybacks).
    4. Distinguish carefully between the requested analysis year, actual year used (if backtracked), and previous comparison year.
    5. Mention data backtracking when data_note indicates it.
    6. When rule-based scores and the trained ML model disagree:
       - Maintain strictly neutral language (e.g., "the two methods are picking up different signals" or "divergence warrants closer review").
       - NEVER claim the ML model "caught what Altman missed", and NEVER claim Altman is "outdated".
       - Treat divergence as an area for deeper analyst investigation, not as one model being right over the other.
    7. Do not claim a company WILL or WILL NOT go bankrupt. Probability is a model output, not a certainty.
    8. Describe the Beneish M-Score as a screening/forensic signal, not as proof of fraud or earnings manipulation.
    9. Do not provide investment advice or buy/sell recommendations.
    10. Structure the assessment concisely in 3-4 structured paragraphs:
        - Paragraph 1: Executive summary & overall standing (stability, value creation, and agreement status).
        - Paragraph 2: Core strengths (profitability drivers, cash quality, returns vs. cost of capital).
        - Paragraph 3: Risks, weaknesses, divergence points, and areas for further due diligence.
    """

    USER_PROMPT = f"""
    Analyze the financial assessment below for preliminary PE/M&A screening.

    Company: {result["ticker"]}
    Analysis Fiscal Year: {result["analysis_year"]}

    --- Supervised ML Classifier ---
    {result.get("bankruptcy")}

    --- Rule-Based Scoring Engine ---
    Altman Z-Score:
    {result.get("altman")}

    Piotroski F-Score:
    {result.get("piotroski")}

    DuPont ROE Breakdown:
    {result.get("dupont")}

    Cash Quality & Conversion:
    {result.get("cash_quality")}

    ROIC vs WACC Value Creation:
    {result.get("roic_wacc")}

    Beneish M-Score (Earnings Quality Screen):
    {result.get("beneish")}

    Write a concise 3-4 paragraph analyst standing report adhering strictly to the system guidelines.
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.3,
    )
    return [response.choices[0].message.content, response.usage]


def sanitize_for_json(obj):
    """
    Recursively converts pandas Timestamps, numpy types, and NaNs to standard JSON-serializable types.
    """
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def map_backend_to_frontend(res, llm_report):
    """
    Maps all raw backend model outputs, rule-based pillars, and LLM text into a unified schema.
    """
    ticker = res["ticker"]
    analysis_year = res["analysis_year"]

    # Fetch company profile
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("longName", ticker)
        sector = info.get("sector", "Unknown Sector")
        business_overview = info.get("longBusinessSummary", "Financial data retrieved from yfinance.")
    except Exception:
        name = ticker
        sector = "Unknown Sector"
        business_overview = "Financial data retrieved from yfinance."

    # 1. Altman Z-Score
    altman = res.get("altman") or {}
    altman_z = altman.get("altman_z_score", 0.0)
    altman_class = altman.get("altman_classification", "Unknown")
    altman_flag = "good" if "Safe" in str(altman_class) else ("neutral" if "Grey" in str(altman_class) else "warn")

    # 2. Piotroski F-Score
    piotroski = res.get("piotroski") or {}
    piotroski_score = piotroski.get("piotroski_score", 0)
    piotroski_class = piotroski.get("classification", "Unknown")
    piotroski_flag = "good" if piotroski_score >= 7 else ("neutral" if piotroski_score >= 4 else "warn")

    # 3. DuPont Breakdown
    dupont = res.get("dupont") or {}
    dupont_roe = dupont.get("roe", 0.0)
    dupont_pm = dupont.get("profit_margin", 0.0)
    dupont_at = dupont.get("asset_turnover", 0.0)
    dupont_em = dupont.get("equity_multiplier", 0.0)
    dupont_driver = dupont.get("primary_driver", "Stable")
    dupont_flag = "good" if dupont_roe >= 0.15 else ("neutral" if dupont_roe >= 0.08 else "warn")

    # 4. Cash Quality
    cash_qual = res.get("cash_quality") or {}
    cash_qual_rating = cash_qual.get("overall_cash_quality", "Unknown")
    fcf = cash_qual.get("free_cash_flow", 0.0)
    cash_conv = cash_qual.get("cash_conversion_ratio", 0.0)
    fcf_conv = cash_qual.get("fcf_conversion_ratio", 0.0)
    cash_flag = "good" if cash_qual_rating == "Strong" else ("neutral" if cash_qual_rating in ["Moderate", "Watch"] else "warn")

    # 5. ROIC vs WACC
    roic_wacc_res = res.get("roic_wacc") or {}
    roic = roic_wacc_res.get("roic", 0.0)
    wacc = roic_wacc_res.get("wacc", 0.0)
    spread = roic_wacc_res.get("economic_spread", 0.0)
    val_creation = roic_wacc_res.get("value_creation", "Unknown")
    roic_flag = "good" if spread > 0.02 else ("neutral" if spread >= -0.01 else "warn")

    # 6. Beneish M-Score
    beneish = res.get("beneish") or {}
    beneish_m = beneish.get("M_score", 0.0)
    beneish_interp = beneish.get("interpretation", "Unknown")
    beneish_flag = "good" if "Unlikely" in str(beneish_interp) else "warn"

    # Rule-Based Composite Confidence (Balanced 6 Pillars)
    # Altman (20%), Piotroski (20%), DuPont (15%), Cash Quality (15%), ROIC/WACC (15%), Beneish (15%)
    altman_pts = 20.0 if altman_flag == "good" else (12.0 if altman_flag == "neutral" else 3.0)
    piotroski_pts = (piotroski_score / 9.0) * 20.0
    dupont_pts = 15.0 if dupont_flag == "good" else (9.0 if dupont_flag == "neutral" else 3.0)
    cash_pts = 15.0 if cash_flag == "good" else (9.0 if cash_flag == "neutral" else 3.0)
    roic_pts = 15.0 if roic_flag == "good" else (8.0 if roic_flag == "neutral" else 2.0)
    beneish_pts = 15.0 if beneish_flag == "good" else 5.0

    composite_confidence = round(altman_pts + piotroski_pts + dupont_pts + cash_pts + roic_pts + beneish_pts)
    composite_confidence = max(5, min(99, composite_confidence))
    confidence_label = "High" if composite_confidence >= 80 else ("Medium" if composite_confidence >= 55 else "Low")


    # Supervised ML Model Distress & Safety Confidence
    bankruptcy = res.get("bankruptcy") or {}
    prob = bankruptcy.get("bankruptcy_probability", 0.0) if bankruptcy else 0.0
    distress_prob = round(prob * 100)
    model_conf = round((1.0 - prob) * 100) if prob < 0.5 else round(prob * 100)
    model_conf_label = "High" if model_conf >= 85 else ("Med-High" if model_conf >= 70 else ("Medium" if model_conf >= 55 else "Low"))

    # Agreement / Divergence Comparison Layer
    gap_points = abs(composite_confidence - model_conf)
    status = "agreement" if gap_points <= 15 else "divergence"

    if status == "agreement":
        explanation = (
            f"Rule-based confidence ({composite_confidence}%) and trained model confidence ({model_conf}%) "
            f"independently converge with a {gap_points}-point gap, confirming consistent financial signals."
        )
    else:
        # Identify divergence driver
        drivers = []
        if dupont_driver == "Leverage-Driven" or (dupont_em and dupont_em > 4.0):
            drivers.append("high financial leverage")
        if cash_flag == "warn" or (fcf_conv is not None and fcf_conv < 0.5):
            drivers.append("cash conversion divergence")
        if roic_flag == "warn":
            drivers.append("spread pressure below cost of capital")
        if not drivers:
            drivers.append("differing sensitivity to balance sheet ratios")

        driver_str = " and ".join(drivers)
        explanation = (
            f"Divergence detected: {gap_points}-point gap between rule-based confidence ({composite_confidence}%) "
            f"and trained model confidence ({model_conf}%). The signals differ primarily due to {driver_str}. "
            f"Analyst review is recommended."
        )

    # Parse LLM Report paragraphs
    paragraphs = [p.strip() for p in llm_report.strip().split("\n\n") if p.strip()]
    valuation_standing = paragraphs[1] if len(paragraphs) > 1 else f"ROIC spread ({spread:.2%}) and return metrics indicate standing."
    conclusion = paragraphs[2] if len(paragraphs) > 2 else (paragraphs[-1] if paragraphs else "Standing analysis complete.")

    # Structured Metrics List for UI
    metrics = [
        {"name": "Piotroski F-Score", "result": f"{piotroski_score} / 9 ({piotroski_class})", "flag": piotroski_flag},
        {"name": "Altman Z-Score", "result": f"{altman_z:.2f} ({altman_class})", "flag": altman_flag},
        {"name": "DuPont ROE", "result": f"{dupont_roe:.1%} ({dupont_driver})", "flag": dupont_flag},
        {"name": "ROIC vs WACC", "result": f"ROIC {roic:.1%} vs WACC {wacc:.1%} (Spread {spread:+.1%})", "flag": roic_flag},
        {"name": "Cash Quality", "result": f"{cash_qual_rating} (FCF Conversion {fcf_conv:.0%})" if fcf_conv is not None else f"{cash_qual_rating}", "flag": cash_flag},
        {"name": "Beneish M-Score", "result": f"{beneish_m:.2f} ({beneish_interp})", "flag": beneish_flag}
    ]

    return sanitize_for_json({
        "id": ticker.lower(),
        "name": name,
        "ticker": ticker,
        "sector": sector,
        "fiscal_year": f"FY{analysis_year}",
        "generated_at": datetime.date.today().strftime("%b %d, %Y"),
        "business_overview": business_overview,
        "rule_based": {
            "composite_confidence": composite_confidence,
            "confidence_label": confidence_label,
            "metrics": metrics
        },
        "trained_model": {
            "distress_probability": distress_prob,
            "model_confidence": model_conf,
            "confidence_label": model_conf_label
        },
        "agreement": {
            "status": status,
            "gap_points": gap_points,
            "explanation": explanation,
            "next_step": "Perform line-item cash flow and debt covenant audit." if status == "divergence" else "Standard due diligence checklist."
        },
        "valuation_standing": valuation_standing,
        "conclusion": conclusion,
        "raw_details": {
            "altman": altman,
            "piotroski": piotroski,
            "beneish": beneish,
            "dupont": dupont,
            "cash_quality": cash_qual,
            "roic_wacc": roic_wacc_res
        },
        "llm_report_paragraphs": paragraphs,
        "llm_report_markdown": llm_report
    })


def generate_json_report(ticker):
    """
    End-to-end execution returning the complete mapped report.
    """
    analysis = analyze_company(ticker)
    if not analysis:
        return None
    report_out = generate_llm_report(analysis)
    return map_backend_to_frontend(analysis, report_out[0])


if __name__ == "__main__":
    import json
    print("\n--- Running Financial Analysis Pipeline for AAPL ---")
    analysis = analyze_company("AAPL")
    print(f"Analysis Year: {analysis['analysis_year']}")
    print(f"Altman: {analysis['altman']['altman_classification']} ({analysis['altman']['altman_z_score']:.2f})")
    print(f"Piotroski: {analysis['piotroski']['classification']} ({analysis['piotroski']['piotroski_score']}/9)")
    print(f"DuPont ROE: {analysis['dupont']['roe']:.2%} (Driver: {analysis['dupont']['primary_driver']})")
    print(f"Cash Quality: {analysis['cash_quality']['overall_cash_quality']} (FCF: ${analysis['cash_quality']['free_cash_flow']:,.0f})")
    print(f"ROIC: {analysis['roic_wacc']['roic']:.2%} vs WACC: {analysis['roic_wacc']['wacc']:.2%} (Spread: {analysis['roic_wacc']['economic_spread']:+.2%})")
    print(f"Beneish M-Score: {analysis['beneish']['M_score']:.2f} ({analysis['beneish']['interpretation']})")
    print(f"ML Distress Probability: {analysis['bankruptcy']['bankruptcy_probability']:.2%}")