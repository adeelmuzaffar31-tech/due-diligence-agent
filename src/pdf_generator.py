"""PDF report generator — renders a validated report dict to a professional PDF.

Uses Jinja2 for templating and WeasyPrint for HTML-to-PDF conversion.
The styling is intentionally clean and corporate; this is a report meant
to be sent to a client, not a flashy demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template
from weasyprint import HTML


REPORT_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  @page { size: A4; margin: 2cm; }
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1a1a1a; line-height: 1.55; font-size: 11pt; }
  .header { border-bottom: 2px solid #1a1a1a; padding-bottom: 18px; margin-bottom: 24px; }
  .header h1 { margin: 0 0 8px; font-size: 22pt; font-weight: 500; }
  .header .subject { font-size: 14pt; color: #555; margin: 0; }
  .meta-row { display: flex; gap: 16px; margin: 14px 0 0; font-size: 9pt; color: #777; }
  .risk-badge { display: inline-block; padding: 5px 14px; border-radius: 20px;
                font-weight: 600; font-size: 10pt; letter-spacing: 0.5px; }
  .risk-LOW       { background: #d4edda; color: #155724; }
  .risk-MEDIUM    { background: #fff3cd; color: #856404; }
  .risk-HIGH      { background: #f8d7da; color: #721c24; }
  .risk-CRITICAL  { background: #f5c6cb; color: #491217; }
  .summary { background: #f8f8f6; padding: 16px 18px; border-radius: 6px;
             border-left: 4px solid #1a1a1a; margin: 0 0 24px; font-size: 11pt; }
  .summary p { margin: 0; }
  h2 { font-size: 14pt; font-weight: 500; margin: 26px 0 12px; color: #1a1a1a;
       border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
  .red-flag { background: #fff5f5; border-left: 3px solid #c00; padding: 8px 12px;
              margin: 6px 0; font-size: 10.5pt; color: #4a1010; }
  .section { margin: 14px 0; padding: 14px 16px;
             border: 1px solid #e6e6e3; border-radius: 6px; }
  .section-head { display: flex; justify-content: space-between;
                  align-items: baseline; margin-bottom: 10px; }
  .section-head h3 { margin: 0; font-size: 12pt; font-weight: 500; }
  .score-pill { font-size: 9pt; padding: 3px 10px; border-radius: 12px;
                background: #efeee8; color: #333; font-weight: 500; }
  .section ul { margin: 4px 0 0 18px; padding: 0; }
  .section li { font-size: 10.5pt; margin: 4px 0; color: #333; }
  .sources { margin: 24px 0 0; font-size: 9pt; color: #555; }
  .sources h3 { font-size: 10pt; font-weight: 500; margin: 0 0 6px; color: #333; }
  .sources li { margin: 2px 0; word-break: break-all; }
  .footer { margin-top: 28px; padding-top: 12px; border-top: 1px solid #e6e6e3;
            font-size: 9pt; color: #888; display: flex; justify-content: space-between; }
</style></head><body>

<div class="header">
  <h1>Due Diligence Report</h1>
  <p class="subject">{{ report.company }}</p>
  <div class="meta-row">
    <span class="risk-badge risk-{{ report.risk_level }}">{{ report.risk_level }} RISK</span>
    <span>Overall risk score: <strong>{{ report.overall_risk_score }}/10</strong></span>
    <span>Confidence: <strong>{{ report.confidence }}</strong></span>
  </div>
</div>

<div class="summary">
  <p>{{ report.executive_summary }}</p>
</div>

{% if report.red_flags %}
<h2>Red flags</h2>
{% for flag in report.red_flags %}
  <div class="red-flag">{{ flag }}</div>
{% endfor %}
{% endif %}

<h2>Detailed assessment</h2>
{% for section_name, section in report.sections.items() %}
<div class="section">
  <div class="section-head">
    <h3>{{ section_name.replace('_', ' ').title() }}</h3>
    <span class="score-pill">{{ section.score }}/10</span>
  </div>
  {% if section.findings %}
  <ul>
    {% for finding in section.findings %}<li>{{ finding }}</li>{% endfor %}
  </ul>
  {% else %}
  <p style="color:#888; font-size:10pt; margin:0;">No specific findings.</p>
  {% endif %}
</div>
{% endfor %}

{% if report.sources %}
<div class="sources">
  <h3>Sources cited</h3>
  <ul>
    {% for src in report.sources %}<li>{{ src }}</li>{% endfor %}
  </ul>
</div>
{% endif %}

<div class="footer">
  <span>Generated {{ report.generated_at[:19].replace('T', ' ') }} UTC</span>
  <span>Due Diligence Agent</span>
</div>

</body></html>"""


def generate_pdf(report: dict[str, Any], output_path: str | Path) -> Path:
    """Render a report dict to a PDF on disk.

    Args:
        report: Validated report from synthesize_report().
        output_path: Where to write the PDF.

    Returns:
        Path to the written PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = Template(REPORT_TEMPLATE).render(report=report)
    HTML(string=html_content).write_pdf(str(output_path))

    return output_path
