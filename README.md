# Standing Report — Frontend Scaffold

Flask + Jinja + Tailwind CSS v4. Renders the AI Company Standing Report in
all three wireframe layout directions (linear scroll, compact cards,
sidebar nav) from one shared data shape and one shared set of Jinja
partials, so switching layouts or plugging in the real backend doesn't
mean rewriting markup.

## Setup

```bash
# Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CSS (Tailwind v4, CSS-first config — no tailwind.config.js needed)
npm install
npm run build:css      # one-off build
# or
npm run watch:css      # rebuild on save while developing
```

## Run

```bash
python run.py
```

Then open `http://127.0.0.1:5000`.

- `/` — portfolio list
- `/report/<company_id>?layout=linear|compact|sidebar` — report detail.
  Every report page has a Linear / Compact / Sidebar switcher at the top
  so you can flip between the three directions for the same company —
  useful when the team is comparing them.

Company ids in the mock data: `mrdn`, `vig`, `crsr`, `hldm`.

## The data contract

Everything the templates render comes from the shape in
`app/data/mock_reports.json` (one object per company: `rule_based`,
`trained_model`, `agreement`, `valuation_standing`, `risk_factors`,
`conclusion`). `load_companies()` in `app/__init__.py` is the single
place that loads this data — right now it reads the JSON fixture; once
the AI backend exists, that function is the only thing that needs to
change (swap it for an API call or DB query returning the same shape).
No template or partial should need to change.

Fields with reference to the schema:

```jsonc
{
  "id": "mrdn",
  "name": "Meridian Foods Corp",
  "ticker": "MRDN",
  "sector": "Packaged Foods",
  "fiscal_year": "FY2025",
  "generated_at": "Jul 31, 2026",
  "business_overview": "...",
  "rule_based": {
    "composite_confidence": 86,
    "confidence_label": "High",
    "metrics": [
      { "name": "Piotroski F-Score", "result": "7 / 9", "flag": "good" }
      // flag: "good" | "neutral" | "warn" — colors the row's status dot
    ]
  },
  "trained_model": {
    "distress_probability": 11,
    "model_confidence": 89,
    "confidence_label": "High"
  },
  "agreement": {
    "status": "agreement",     // or "divergence"
    "gap_points": 3,
    "explanation": "...",
    "next_step": null          // shown only for divergence, otherwise null
  },
  "valuation_standing": "...",
  "risk_factors": ["...", "..."],
  "conclusion": "..."
}
```

If the real backend's shape ends up differing (extra fields, renamed
keys, a metric list of different length per company — already handled),
adjust `load_companies()` to map their output into this shape rather
than touching the templates.

## Structure

```
app/
  __init__.py              # app factory + routes + load_companies()
  data/mock_reports.json   # fixture — swap for real backend later
  templates/
    base.html
    portfolio.html          # company list
    report_linear.html      # layout 1a
    report_compact.html     # layout 1b
    report_sidebar.html     # layout 1c
    partials/
      _badges.html          # confidence_badge(), status_pill()
      _status_banner.html   # status_banner() — agreement/divergence box
      _score_table.html     # score_table() — rule-based metrics table
      _risk_factors.html    # risk_factors()
      _layout_switcher.html # Linear/Compact/Sidebar links
  static/css/
    input.css               # Tailwind v4 theme tokens (@theme block)
    output.css               # built (gitignored in a real repo)
```

## Notes

- Colors, spacing, and copy follow the wireframes as given rather than
  reinterpreting them — the palette lives in the `@theme` block in
  `input.css` if it needs to shift later.
- Missing/pending fields: wrap access with `default` filters (e.g.
  `{{ company.trained_model.model_confidence | default('—') }}`) if a
  report can land with the trained-model pass still running.
