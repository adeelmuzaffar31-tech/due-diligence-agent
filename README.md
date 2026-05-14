# Due Diligence Agent

An AI agent that researches any company across multiple data sources and produces a structured PDF risk report — automatically.

Give it a company name. It fans out to web search, news sources, and public filings, collects evidence, reasons about red flags, and generates a professional PDF report with risk scores, executive summary, and source citations. What takes a human analyst 3–4 hours, the agent completes in under 5 minutes.

**Model-agnostic.** Built with a clean provider abstraction — runs on both **Claude (Anthropic)** and **GPT-4 (OpenAI)** with a single environment variable switch. Same agent loop, same tool definitions, same output schema.

**Built with production discipline.** This isn't a notebook demo — it's structured the way a real backend service should be: clean module boundaries, validated I/O at every boundary (Pydantic), provider-agnostic LLM calls, capped context windows, and graceful tool error handling. Drawn from 3+ years building production Node.js APIs on AWS Lambda, API Gateway, and PostgreSQL.

---

## What it does

```
User: "Research Acme Corporation for due diligence"

Agent:
  → Searches news, press, lawsuits         [Tavily API]
  → Pulls public filings                   [SEC EDGAR]
  → Checks sanctions lists                 [OFAC]
  → Reasons across all evidence            [Claude Sonnet OR GPT-4]
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
│  Orchestrator (Claude tool-use OR GPT-4         │
│  function-calling — unified via provider layer) │
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
git clone https://github.com/adeelmuzaffar31-tech/due-diligence-agent.git
cd due-diligence-agent

pip install -r requirements.txt

cp .env.example .env
# Edit .env: pick your provider, add your API key(s) and Tavily key

# Run with Claude (default)
python -m src.main "Acme Technologies"

# Run with GPT-4 instead — same agent, different model
LLM_PROVIDER=openai python -m src.main "Acme Technologies"
```

The PDF appears in `reports/acme_technologies_diligence.pdf`.

## Switching between providers

The agent works identically with either model. Set one environment variable:

```bash
# Use Claude (default)
LLM_PROVIDER=anthropic

# Use GPT-4
LLM_PROVIDER=openai
```

You only need an API key for whichever provider you're using. Both providers expose tool-use (Anthropic) / function-calling (OpenAI), and the `llm_provider.py` module normalizes the differences so the rest of the codebase stays clean.

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

## Tech stack

**AI layer**
- LLMs: Claude Sonnet (Anthropic) or GPT-4 (OpenAI) — interchangeable via abstraction
- Tool use / function calling for autonomous research
- Pydantic schemas for validated structured output
- Two-pass architecture: parallel research → synthesis

**Backend layer**
- Python 3.10+
- Async HTTP with httpx
- WeasyPrint + Jinja2 for professional PDF rendering
- Modular design ready for FastAPI / Celery deployment

**Data sources**
- Tavily for web search (clean text extraction)
- SEC EDGAR for public filings (free)
- US Treasury OFAC for sanctions screening (free)

## Cost per report

| Component | Claude Sonnet | GPT-4o |
|---|---|---|
| Research + synthesis | $0.15–$0.30 | $0.10–$0.25 |
| Tavily search (~5 calls) | $0.05 | $0.05 |
| EDGAR, OFAC | Free | Free |
| **Total** | **~$0.20–$0.35** | **~$0.15–$0.30** |

## Project structure

```
due-diligence-agent/
├── src/
│   ├── llm_provider.py   # Provider abstraction (Claude + GPT)
│   ├── agent.py          # Orchestrator loop
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

- [ ] FastAPI wrapper for REST endpoint deployment
- [ ] Docker container for one-command deployment
- [ ] Token usage + cost tracking per run
- [ ] Retry logic with exponential backoff for API failures
- [ ] Caching layer for repeated company lookups
- [ ] Add Crunchbase / Proxycurl for startup intelligence
- [ ] Court records search (CourtListener API)
- [ ] Webhook support for async report delivery

## License

MIT — see [LICENSE](LICENSE)

## About the author

Built by [Adeel Muzaffar](https://www.linkedin.com/in/adeel-muzaffar-b61624172) — AI Agent Engineer with 3+ years building production backend systems.

**Background**: Node.js / Express / NestJS APIs in production at Meissasoft, KYC state machine design on PostgreSQL, AWS Lambda + API Gateway + CloudFront at scale. Now specializing in AI agent architecture — applying that backend engineering discipline to autonomous systems.

**Stack**: Claude API · OpenAI / GPT-4 · LangGraph · LangChain · Python · FastAPI · Node.js · Express · NestJS · PostgreSQL · MongoDB · AWS · RAG

Available for remote projects — DM on LinkedIn.
