import json
from pathlib import Path

from flask import Flask, render_template, request, abort

DATA_PATH = Path(__file__).parent / "data" / "mock_reports.json"

LAYOUT_TEMPLATES = {
    "linear": "report_linear.html",
    "compact": "report_compact.html",
    "sidebar": "report_sidebar.html",
}


def load_companies():
    """
    Swap this function's body for a real API/DB call once the AI backend
    is ready. Everything downstream (templates, partials) only depends on
    the shape defined in mock_reports.json, so nothing else has to change.
    """
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["companies"]


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

    return app
