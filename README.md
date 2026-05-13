# Due Diligence Agent

An AI agent that researches any company across multiple data sources and produces a structured PDF risk report — automatically.

Give it a company name. It fans out to web search, news sources, and public filings, collects evidence, reasons about red flags, and generates a professional PDF report with risk scores, executive summary, and source citations. What takes a human analyst 3–4 hours, the agent completes in under 5 minutes.

Built with Claude's tool-use API and a parallel research → single synthesis pass architecture.

---

## What it does

```
User: "Research Acme Corporation for due diligence"

Agent:
  → Searches news, press, lawsuits         [Tavily API]
  → Pulls public filings                   [SEC EDGAR]
  → Checks sanctions lists                 [OFAC]
  → Reasons across all evidence            [Claude Sonnet]
  → Generates structured report            [Pydantic schema]
  → Renders professional PDF               [WeasyPrint]

Output: acme_corporation_diligence.pdf
        (5 minutes, ~$0.30 in API costs)
```

## Architecture

The agent uses a two-pass architecture: **research** then **synthesis**. The model never reasons and researches in the same call — keeping them separated produces cleaner outputs and lower costs.

```
┌─────────────────────────────────────────────────┐
│  User input: company name                       │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  Orchestrator (Claude Sonnet + tool use)        │
│  Plans research, calls tools in parallel        │
└──────┬──────────┬──────────┬──────────┬─────────┘
       ▼          ▼          ▼          ▼
   ┌───────┐ ┌───────┐  ┌───────┐  ┌───────┐
   │  Web  │ │  SEC  │  │ OFAC  │  │  News │
   │search │ │ EDGAR │  │  list │  │ feeds │
   └───┬───┘ └───┬───┘  └───┬───┘  └───┬───┘
       └─────────┴──────────┴──────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  Synthesizer (separate LLM call)                │
│  Scores risks, extracts findings, cites sources │
│  Returns structured JSON                        │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  PDF generator (WeasyPrint + Jinja2)            │
└─────────────────────────────────────────────────┘
```

## Quick start

```bash
# Clone
git clone https://github.com/adeelmuzaffar31-tech/due-diligence-agent.git
cd due-diligence-agent

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and TAVILY_API_KEY

# Run on a real company
python -m src.main "Acme Technologies"
```

The PDF appears in `reports/acme_technologies_diligence.pdf`.

## What's in a report

Every report contains:

- **Executive summary** — 2–3 sentence plain-English overview
- **Overall risk score** — 1–10 scale with risk level (LOW / MEDIUM / HIGH / CRITICAL)
- **Red flags** — specific concerns with source citations
- **Section breakdowns**:
  - Legal & regulatory
  - Financial health
  - Leadership team
  - Market position
  - Reputational risk
- **Sources** — every URL the agent cited
- **Confidence level** — how reliable the report is given available data

See `examples/sample_report.pdf` for a real generated report.

## Tech stack

- **LLM**: Claude Sonnet (Anthropic API) with native tool use
- **Web search**: Tavily API (clean text extraction, not raw HTML)
- **SEC filings**: EDGAR full-text search (free)
- **Sanctions**: US Treasury OFAC API (free)
- **PDF rendering**: WeasyPrint + Jinja2 templates
- **Validation**: Pydantic for structured outputs
- **Language**: Python 3.10+

## Cost per report

| Component | Cost |
|---|---|
| Claude API (research + synthesis) | $0.15 – $0.30 |
| Tavily search calls (~5 per report) | $0.05 |
| EDGAR, OFAC | Free |
| **Total** | **~$0.20 – $0.35** |

At 30:1 margins, this is a high-leverage project to build on top of.

## Project structure

```
due-diligence-agent/
├── src/
│   ├── agent.py          # Orchestrator loop with tool-use
│   ├── tools.py          # Web search, EDGAR, OFAC implementations
│   ├── synthesizer.py    # Second-pass synthesis to structured JSON
│   ├── pdf_generator.py  # PDF rendering with WeasyPrint
│   └── main.py           # CLI entry point
├── examples/             # Sample reports
├── tests/                # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap

- [ ] Add Crunchbase / Proxycurl integration for startup intelligence
- [ ] Court records search (CourtListener API)
- [ ] LinkedIn team analysis
- [ ] FastAPI wrapper for REST endpoint deployment
- [ ] Webhook support for async report delivery

## License

MIT — see [LICENSE](LICENSE)

## About

Built by [Adeel Muzaffar](https://www.linkedin.com/in/adeel-muzaffar-b61624172) — AI Agent Engineer specialising in autonomous agents that automate research and operations workflows. Available for projects.
