"""Smoke tests for the agent.

These don't make real API calls — they verify imports and structure work.
Run with: python -m pytest tests/
"""

from src.pdf_generator import generate_pdf
from src.synthesizer import DiligenceReport
from src.tools import TOOL_DEFINITIONS, execute_tool


def test_tool_definitions_well_formed():
    """Every tool definition should have the required Anthropic API fields."""
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_unknown_tool_returns_error():
    """Dispatcher should fail gracefully on unknown tool names."""
    result = execute_tool("nonexistent_tool", {})
    assert "error" in result


def test_report_schema_validates():
    """The Pydantic schema should accept a well-formed report."""
    report = DiligenceReport(
        company="Test Corp",
        overall_risk_score=5,
        risk_level="MEDIUM",
        executive_summary="Test summary for schema validation.",
        red_flags=["Test flag"],
        sections={
            "legal_regulatory":  {"score": 5, "findings": ["test"]},
            "financial_health":  {"score": 4, "findings": []},
            "leadership_team":   {"score": 3, "findings": []},
            "market_position":   {"score": 6, "findings": []},
            "reputational_risk": {"score": 5, "findings": []},
        },
        sources=["https://example.com"],
        generated_at="2026-01-01T00:00:00Z",
        confidence="MEDIUM",
    )
    assert report.company == "Test Corp"
    assert report.overall_risk_score == 5


def test_pdf_generation(tmp_path):
    """PDF generation should produce a non-empty file."""
    fake_report = {
        "company": "Test Corp",
        "overall_risk_score": 5,
        "risk_level": "MEDIUM",
        "executive_summary": "Test summary.",
        "red_flags": ["Test flag"],
        "sections": {
            "legal_regulatory":  {"score": 5, "findings": ["test finding"]},
            "financial_health":  {"score": 4, "findings": []},
            "leadership_team":   {"score": 3, "findings": []},
            "market_position":   {"score": 6, "findings": []},
            "reputational_risk": {"score": 5, "findings": []},
        },
        "sources": ["https://example.com"],
        "generated_at": "2026-01-01T00:00:00Z",
        "confidence": "MEDIUM",
    }
    output = tmp_path / "test_report.pdf"
    result_path = generate_pdf(fake_report, output)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000
