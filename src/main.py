"""CLI entry point — run a full due diligence pipeline on a single company.

Usage:
    python -m src.main "Company Name"
    python -m src.main "Company Name" --output reports/my_report.pdf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_research_agent
from .pdf_generator import generate_pdf
from .synthesizer import synthesize_report


def slugify(name: str) -> str:
    """Convert a company name into a safe filename slug."""
    safe = "".join(c if c.isalnum() else "_" for c in name.lower())
    return "_".join(filter(None, safe.split("_")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered due diligence report for any company."
    )
    parser.add_argument("company", help="Company name to research")
    parser.add_argument(
        "--output",
        "-o",
        help="Output PDF path (default: reports/<slug>_diligence.pdf)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    load_dotenv()
    verbose = not args.quiet

    output_path = (
        Path(args.output)
        if args.output
        else Path("reports") / f"{slugify(args.company)}_diligence.pdf"
    )

    started = time.time()

    if verbose:
        print(f"\n→ Researching: {args.company}")
    research = run_research_agent(args.company, verbose=verbose)

    if verbose:
        print(f"\n→ Synthesizing report from {len(research)} pieces of evidence...")
    report = synthesize_report(args.company, research)

    if verbose:
        print(f"\n→ Generating PDF at {output_path}...")
    pdf_path = generate_pdf(report, output_path)

    elapsed = time.time() - started
    if verbose:
        print(f"\n✓ Done in {elapsed:.1f}s")
        print(f"  Risk level:  {report['risk_level']}")
        print(f"  Risk score:  {report['overall_risk_score']}/10")
        print(f"  Red flags:   {len(report['red_flags'])}")
        print(f"  Report PDF:  {pdf_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
