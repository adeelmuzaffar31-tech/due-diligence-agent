"""Synthesizer — turns raw research evidence into a structured risk report.

A separate LLM call from the research agent. Takes all gathered evidence and
produces validated JSON via Pydantic schemas. This separation keeps the
research agent focused and the synthesis pass cheap and deterministic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# Report schema — what every report must contain
# ============================================================

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]


class SectionFinding(BaseModel):
    score: int = Field(ge=1, le=10, description="Risk score for this section, 1=low risk, 10=critical")
    findings: list[str] = Field(default_factory=list, description="Specific findings with sources")


class DiligenceReport(BaseModel):
    company: str
    overall_risk_score: int = Field(ge=1, le=10)
    risk_level: RiskLevel
    executive_summary: str = Field(description="2-3 sentence plain-English summary")
    red_flags: list[str] = Field(default_factory=list)
    sections: dict[str, SectionFinding]
    sources: list[str] = Field(default_factory=list)
    generated_at: str
    confidence: Confidence


SYNTHESIZER_PROMPT = """You are a senior due diligence analyst. You have been given raw research evidence about a company. Your job is to synthesize it into a structured risk report.

Output ONLY valid JSON in this exact schema. No preamble, no markdown fences, no commentary:

{
  "company": "string",
  "overall_risk_score": <int 1-10>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "executive_summary": "<2-3 sentences in plain English>",
  "red_flags": ["specific concern 1", "specific concern 2", ...],
  "sections": {
    "legal_regulatory": {"score": <int>, "findings": ["..."]},
    "financial_health": {"score": <int>, "findings": ["..."]},
    "leadership_team":   {"score": <int>, "findings": ["..."]},
    "market_position":   {"score": <int>, "findings": ["..."]},
    "reputational_risk": {"score": <int>, "findings": ["..."]}
  },
  "sources": ["url1", "url2", ...],
  "generated_at": "<ISO timestamp>",
  "confidence": "LOW" | "MEDIUM" | "HIGH"
}

Scoring guide: 1-3 = low risk, 4-6 = moderate, 7-8 = high, 9-10 = critical.
Risk level should align with overall_risk_score: 1-3=LOW, 4-5=MEDIUM, 6-8=HIGH, 9-10=CRITICAL.

Confidence reflects how reliable your assessment is given the evidence available:
- HIGH: comprehensive evidence across all sections
- MEDIUM: some sections under-evidenced
- LOW: limited or contradictory evidence

If evidence is sparse for a section, give it a moderate score (5) and note the gap in findings.
Cite specific URLs from the evidence in the sources list."""


def synthesize_report(
    company_name: str,
    research_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the synthesis LLM call and return a validated report dict.

    Args:
        company_name: Subject of the report.
        research_findings: Output from run_research_agent().

    Returns:
        Validated report as a dict, ready for PDF rendering.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    # Build a compact evidence dump — JSON-encoded, capped for context
    evidence_blob = json.dumps(research_findings, indent=2)[:40000]

    user_prompt = (
        f"Subject company: {company_name}\n"
        f"Current timestamp (use for generated_at): {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Research evidence:\n{evidence_blob}\n\n"
        "Produce the structured JSON report now. JSON only."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYNTHESIZER_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract text from response
    raw_text = "".join(
        block.text for block in response.content if hasattr(block, "text")
    ).strip()

    # Strip markdown fences if the model included them despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Synthesizer returned non-JSON output:\n{raw_text[:500]}"
        ) from exc

    try:
        validated = DiligenceReport(**parsed)
    except ValidationError as exc:
        raise RuntimeError(
            f"Synthesizer output failed schema validation: {exc}"
        ) from exc

    return validated.model_dump()
