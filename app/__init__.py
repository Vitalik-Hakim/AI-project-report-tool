import json
import datetime
import sys
from pathlib import Path
from flask import Flask, render_template, request, abort, redirect, url_for, jsonify
from dotenv import load_dotenv

# Add backend directory to sys.path with fallback resolution
possible_backend_paths = [
    Path(__file__).resolve().parent.parent.parent / "intro-to-ai-project",
    Path(__file__).resolve().parent.parent / "intro-to-ai-project",
    Path("/Users/vincentchanayire/Downloads/intro-to-ai-project"),
]

for p in possible_backend_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))
        load_dotenv(p / ".env")
        break

DATA_PATH = Path(__file__).parent / "testdata" / "reports.json"
MOCK_DATA_PATH = Path(__file__).parent / "testdata" / "mock_reports.json"

LAYOUT_TEMPLATES = {
    "linear": "report_linear.html",
    "compact": "report_compact.html",
    "sidebar": "report_sidebar.html",
}


def load_companies():
    """
    Load real companies reports database, falling back to mock data if not generated yet.
    """
    path = DATA_PATH if DATA_PATH.exists() else MOCK_DATA_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["companies"]


def save_company(company_data):
    """
    Save or update a company report in the real reports database.
    """
    if DATA_PATH.exists():
        try:
            with open(DATA_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"companies": []}
    else:
        # Start by seeding with existing mock reports so they aren't lost
        try:
            with open(MOCK_DATA_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"companies": []}

    companies = data.setdefault("companies", [])
    
    # Update if exists, otherwise append
    existing_idx = next((i for i, c in enumerate(companies) if c["id"] == company_data["id"]), None)
    if existing_idx is not None:
        companies[existing_idx] = company_data
    else:
        companies.append(company_data)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)




def create_app():
    app = Flask(__name__)

    @app.route("/")
    def portfolio():
        companies = load_companies()
        return render_template("portfolio.html", companies=companies)

    @app.route("/report/<company_id>")
    def report(company_id):
        companies = load_companies()
        company = next((c for c in companies if c["id"] == company_id), None)
        if company is None:
            abort(404)

        layout = request.args.get("layout", "linear")
        template_name = LAYOUT_TEMPLATES.get(layout, "report_linear.html")
        return render_template(
            template_name,
            company=company,
            layout=layout,
            all_companies=companies,
        )

    @app.route("/analyze", methods=["POST"])
    def analyze():
        ticker = request.form.get("ticker", "").strip().upper()
        if not ticker:
            return redirect(url_for("portfolio"))
        return render_template("loading.html", ticker=ticker)

    @app.route("/api/analyze/<ticker>")
    def api_analyze(ticker):
        ticker = ticker.strip().upper()
        if not ticker:
            return jsonify({"status": "error", "message": "Ticker cannot be empty."}), 400
            
        try:
            from financial_analysis import generate_json_report
            
            # Run the unified backend report pipeline
            mapped = generate_json_report(ticker)
            if not mapped:
                return jsonify({"status": "error", "message": f"Could not generate report for {ticker}."}), 400
                
            # Save mapped report to local db
            save_company(mapped)
            
            return jsonify({"status": "success", "company_id": mapped["id"]})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/methodology")
    def methodology():
        return render_template("methodology.html")

    return app
