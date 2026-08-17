# Standing Report Dashboard
### Interactive Financial Analysis & Due Diligence UI

**CS 254: Introduction to Artificial Intelligence — Final Project**  
*Ashesi University, Department of Computer Science — May–August 2026*

---

## 1. Project Overview

The **Standing Report Dashboard** is a Flask + Jinja + Tailwind CSS v4 web application designed for private equity (PE) and mergers & acquisitions (M&A) target screening. It renders multi-pillar deterministic financial metrics, supervised ML distress probabilities, and AI-generated standing briefs across three distinct viewing layouts (Linear Scroll, Compact Cards, and Sidebar Navigation) using a unified data contract.

---

## 2. Team Members

* **Abdul Hakim Aremeyaw** — Software Engineering, Frontend Architecture & Full-Stack Deployment
* **Haris Tiyumtaba Issah** — Data Engineering, Feature Selection & Supervised ML Classifier
* **Samira Ewura-Esi Donkoh** — Financial Framework Research, Ethics Audit & Final Report Lead
* **Vincent Adijore Chanayire** — Rule-Based Financial Engines, Pipeline Integration & Presentation Lead

---

## 3. Key Features

* **Real-Time Live Analysis (`/analyze`)**: Enter any public ticker symbol (e.g. `AAPL`, `MSFT`, `CAT`, `XOM`) to trigger the live Python extraction and scoring pipeline.
* **Interactive 6-Pillar Calculators**:
  1. **DuPont 3-Step ROE Decomposition** (Net Margin &times; Asset Turnover &times; Equity Multiplier with operational vs. leverage driver detection).
  2. **Cash Quality Engine** (Operating Cash Flow, CapEx, Free Cash Flow, Cash Conversion $OCF/NI$, and FCF Conversion trend).
  3. **ROIC vs. WACC Economic Spread** (NOPAT, Invested Capital, CAPM Cost of Capital, Capital Structure weights, and Value Creation meter).
  4. **Piotroski F-Score 9-Rule Checklist** (Pass/fail breakdown across profitability, leverage, and efficiency).
  5. **Altman Z-Score Solvency Meter** (Safe, Grey, and Distress zone gauge).
  6. **Beneish M-Score Earnings Quality Screen** (8 forensic manipulation risk indices).
* **Calibrated ML Distress Read**: Displays calibrated Random Forest distress probability ($P(\text{distress})$) alongside composite rule confidence.
* **Agreement & Divergence Banner**: Highlights consensus or divergence points with neutral, non-judgmental explanations and recommended diligence next steps.
* **Layout Switcher**: Dynamically toggle between **Linear**, **Compact**, and **Sidebar** views for any company standing report.

---

## 4. Setup & Installation

### Step 1: Install Python Dependencies
```bash
# In the AI-project-report-tool directory
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Build Tailwind CSS
```bash
npm install
npm run build:css      # Compile CSS to app/static/css/output.css
# or for live watch during development:
npm run watch:css
```

---

## 5. Running the Application

```bash
python run.py
```

Then open your browser at **`http://127.0.0.1:5001`**.

### Available Routes:
* **`/`** — Portfolio overview and company search bar.
* **`/report/<company_id>?layout=linear|compact|sidebar`** — Detailed standing report with interactive layout switcher.
* **`/methodology`** — Comprehensive architectural and mathematical methodology guide.
* **`/api/analyze/<ticker>`** — JSON API endpoint running the live end-to-end backend analysis pipeline.

---

## 6. Directory Structure

```
AI-project-report-tool/
├── app/
│   ├── __init__.py              # Flask app factory, routing, & backend integration
│   ├── data/
│   │   ├── mock_reports.json    # Seed database fixture
│   │   └── reports.json         # Persisted live reports database
│   ├── static/
│   │   └── css/
│   │       ├── input.css        # Tailwind CSS v4 design tokens & base themes
│   │       └── output.css       # Compiled, minified CSS stylesheet
│   └── templates/
│       ├── base.html            # Core layout wrapper
│       ├── portfolio.html       # Portfolio listing and ticker search card
│       ├── report_linear.html   # Layout 1: Linear scroll view
│       ├── report_compact.html  # Layout 2: Compact summary cards
│       ├── report_sidebar.html  # Layout 3: Left-navigation sidebar view
│       ├── loading.html         # Live asynchronous analysis loading screen
│       ├── methodology.html     # Technical methodology & formulas guide
│       └── partials/
│           ├── _badges.html          # Confidence & status badge macros
│           ├── _details_tabs.html    # 6 interactive calculator panel macros
│           ├── _layout_switcher.html # View switcher widget
│           ├── _risk_factors.html    # Risk factor list macro
│           ├── _score_table.html     # Summary metrics table macro
│           └── _status_banner.html   # Agreement / divergence banner macro
├── package.json                 # Tailwind CSS v4 build script config
├── package-lock.json            # Node dependency lockfile
├── requirements.txt             # Python dependencies
├── run.py                       # Application entry point
└── README.md                    # Documentation & setup guide
```

---

## 7. Academic Integrity & Citations

Developed as part of the CS 254 Introduction to AI curriculum at Ashesi University. All metrics are cited from original foundational literature (Altman 1968, Piotroski 2000, Beneish 1999, DuPont 1961, Sharpe 1964).
