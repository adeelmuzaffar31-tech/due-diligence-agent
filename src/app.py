"""Streamlit web UI for the Due Diligence Agent.

Run with:
    streamlit run src/app.py

Provides a clean web interface for non-technical users to:
  - Enter any company name
  - Pick LLM provider and model on the fly
  - Watch live agent progress
  - Download the generated PDF report
  - Browse the structured findings inline
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# Streamlit runs this file as a top-level script, so relative imports fail.
# Put the project root on sys.path so the `src` package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.agent import run_research_agent
from src.pdf_generator import generate_pdf
from src.synthesizer import synthesize_report

load_dotenv()


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Due Diligence Agent",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# Sidebar — provider configuration
# ============================================================

with st.sidebar:
    st.markdown("### Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["anthropic", "openai"],
        index=0,
        help="Both providers produce equivalent reports. Claude is recommended for tool-use reliability.",
    )

    if provider == "anthropic":
        model = st.selectbox(
            "Model",
            options=[
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-5-20251001",
            ],
            index=0,
        )
        api_key_env = "ANTHROPIC_API_KEY"
        os.environ["CLAUDE_MODEL"] = model
    else:
        model = st.selectbox(
            "Model",
            options=[
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-5-mini",
                "gpt-5",
                "o3-mini",
                "o4-mini",
            ],
            index=0,
            help="gpt-4o-mini is cheapest. gpt-5 is most expensive but most capable.",
        )
        api_key_env = "OPENAI_API_KEY"
        os.environ["OPENAI_MODEL"] = model

    os.environ["LLM_PROVIDER"] = provider

    # API key status indicators
    st.markdown("### API keys")
    anth_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    oai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    tav_ok = bool(os.environ.get("TAVILY_API_KEY"))

    st.markdown(
        f"- Anthropic: {'✅ set' if anth_ok else '❌ missing'}\n"
        f"- OpenAI: {'✅ set' if oai_ok else '❌ missing'}\n"
        f"- Tavily: {'✅ set' if tav_ok else '❌ missing'}"
    )

    if not os.environ.get(api_key_env):
        st.error(f"⚠️ {api_key_env} not found in .env file")

    if not tav_ok:
        st.error("⚠️ TAVILY_API_KEY required for web search")

    st.markdown("---")
    st.markdown(
        "### About\n"
        "AI agent that researches any company across news, filings, and "
        "sanctions lists, then produces a structured PDF risk report. "
        "Built with Claude/GPT tool-use and Python.\n\n"
        "[GitHub repo](https://github.com/adeelmuzaffar31-tech/due-diligence-agent) · "
        "[LinkedIn](https://www.linkedin.com/in/adeel-muzaffar-b61624172)"
    )


# ============================================================
# Main panel
# ============================================================

st.title("🔍 Due Diligence Agent")
st.caption(
    "Enter any company name — the agent will research news, lawsuits, filings, "
    "and sanctions, then generate a professional PDF risk report."
)

# Input row
col_input, col_button = st.columns([3, 1])
with col_input:
    company_name = st.text_input(
        "Company name",
        placeholder="e.g. Tesla, Stripe, Careem",
        label_visibility="collapsed",
    )
with col_button:
    run_button = st.button("Research", type="primary", use_container_width=True)


# ============================================================
# Session state — keep the last report in memory
# ============================================================

if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None
if "last_company" not in st.session_state:
    st.session_state.last_company = None
if "last_research" not in st.session_state:
    st.session_state.last_research = None


# ============================================================
# Run pipeline
# ============================================================

if run_button and company_name.strip():
    # Pre-flight checks
    if not os.environ.get(api_key_env):
        st.error(f"Cannot run: {api_key_env} is not set. Add it to your .env file.")
        st.stop()
    if not os.environ.get("TAVILY_API_KEY"):
        st.error("Cannot run: TAVILY_API_KEY is not set. Get a free key at tavily.com")
        st.stop()

    with st.status(f"Researching {company_name}...", expanded=True) as status:
        started = time.time()

        # Phase 1: Research
        st.write(f"🤖 **Agent using:** {provider} ({model})")
        st.write("🔎 Phase 1/3: Gathering evidence...")

        try:
            research = run_research_agent(company_name.strip(), verbose=False)
            st.write(f"   ✓ Collected {len(research)} pieces of evidence")
        except Exception as exc:
            status.update(label="Research failed", state="error")
            st.error(f"Research agent failed: {exc}")
            st.stop()

        # Phase 2: Synthesis
        st.write("🧠 Phase 2/3: Synthesizing structured report...")
        try:
            report = synthesize_report(company_name.strip(), research)
            st.write(f"   ✓ Risk level: **{report['risk_level']}** "
                     f"(score {report['overall_risk_score']}/10)")
        except Exception as exc:
            status.update(label="Synthesis failed", state="error")
            st.error(f"Synthesizer failed: {exc}")
            st.stop()

        # Phase 3: PDF generation
        st.write("📄 Phase 3/3: Generating PDF report...")
        slug = "".join(c if c.isalnum() else "_" for c in company_name.lower())
        output_path = Path("reports") / f"{slug}_diligence.pdf"
        try:
            pdf_path = generate_pdf(report, output_path)
            pdf_bytes = pdf_path.read_bytes()
            st.write(f"   ✓ PDF saved ({len(pdf_bytes) // 1024} KB)")
        except Exception as exc:
            status.update(label="PDF generation failed", state="error")
            st.error(f"PDF generator failed: {exc}")
            st.stop()

        elapsed = time.time() - started
        status.update(
            label=f"✅ Report generated in {elapsed:.1f}s",
            state="complete",
            expanded=False,
        )

    # Store in session
    st.session_state.last_report = report
    st.session_state.last_pdf_bytes = pdf_bytes
    st.session_state.last_company = company_name.strip()
    st.session_state.last_research = research


# ============================================================
# Display results if we have any
# ============================================================

if st.session_state.last_report:
    report = st.session_state.last_report
    company = st.session_state.last_company

    st.markdown("---")

    # Header row with metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    risk_emoji = {
        "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"
    }.get(report["risk_level"], "⚪")
    with col_a:
        st.metric("Risk level", f"{risk_emoji} {report['risk_level']}")
    with col_b:
        st.metric("Risk score", f"{report['overall_risk_score']}/10")
    with col_c:
        st.metric("Red flags", len(report["red_flags"]))
    with col_d:
        st.metric("Confidence", report["confidence"])

    # Download button
    st.download_button(
        label="📥 Download PDF report",
        data=st.session_state.last_pdf_bytes,
        file_name=f"{company.lower().replace(' ', '_')}_diligence.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    # Executive summary
    st.markdown("### Executive summary")
    st.info(report["executive_summary"])

    # Red flags
    if report["red_flags"]:
        st.markdown("### 🚩 Red flags")
        for flag in report["red_flags"]:
            st.markdown(f"- {flag}")

    # Detailed sections
    st.markdown("### Detailed assessment")
    for section_name, section in report["sections"].items():
        title = section_name.replace("_", " ").title()
        score = section["score"]
        section_emoji = "🔴" if score >= 7 else "🟠" if score >= 5 else "🟢"

        with st.expander(f"{section_emoji} {title} — {score}/10", expanded=False):
            if section["findings"]:
                for finding in section["findings"]:
                    st.markdown(f"- {finding}")
            else:
                st.caption("No specific findings for this section.")

    # Sources
    if report["sources"]:
        with st.expander(f"📚 Sources ({len(report['sources'])} cited)"):
            for src in report["sources"]:
                st.markdown(f"- [{src}]({src})")

    # Raw research evidence (collapsed by default — for transparency)
    if st.session_state.last_research:
        with st.expander(f"🔬 Raw research data ({len(st.session_state.last_research)} tool calls)"):
            st.caption(
                "Every tool call the agent made during research, with full results. "
                "Useful for debugging or verifying the agent's evidence chain."
            )
            for i, finding in enumerate(st.session_state.last_research, 1):
                st.markdown(f"**{i}. `{finding['tool']}`**")
                st.json(finding["input"])

elif not run_button:
    # Empty state — show instructions
    st.markdown("---")
    st.markdown(
        "### How it works\n"
        "1. **Enter a company name** in the box above\n"
        "2. The agent **searches news, lawsuits, and filings** in parallel\n"
        "3. A second LLM pass **synthesizes findings into risk scores**\n"
        "4. You get a **professional PDF report** in under 5 minutes\n\n"
        "**Try these examples:** Tesla, Stripe, Theranos, Careem, Airlift"
    )
