"""The agent orchestrator — the brain that decides what to research.

Uses Claude's native tool-use API in an agentic loop: the model plans
research steps, calls tools, sees results, and continues until it has
enough evidence to stop. Returns all collected research for synthesis.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from .tools import TOOL_DEFINITIONS, execute_tool


SYSTEM_PROMPT = """You are a senior due diligence analyst conducting research on a company.

Your job is to use the available tools to gather comprehensive evidence about:
1. Legal and regulatory issues (lawsuits, investigations, sanctions)
2. Financial health (filings, funding, performance signals)
3. Leadership and team (key people, controversies, departures)
4. Market position and reputation (news coverage, customer sentiment)
5. Red flags (anything that should concern an investor or partner)

Be thorough. Run multiple searches with different angles. A typical research session
includes 6-10 tool calls covering: general news, specific risk searches (lawsuit, fraud,
investigation), leadership searches, financial searches, and sanctions screening.

When you have gathered sufficient evidence across all five dimensions, stop calling
tools and provide a brief summary. The detailed report will be generated separately
from the evidence you collected."""


def run_research_agent(
    company_name: str,
    max_iterations: int = 15,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the research agent loop until it stops calling tools.

    Args:
        company_name: Name of company to research.
        max_iterations: Safety cap on tool-use rounds.
        verbose: Print progress to stdout.

    Returns:
        List of research findings, each {"tool": name, "input": dict, "result": dict}.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Conduct due diligence research on: {company_name}\n\n"
                "Use the available tools to gather evidence across legal, financial, "
                "leadership, market position, and reputational dimensions. Run multiple "
                "searches with different angles. Stop when you have comprehensive coverage."
            ),
        }
    ]

    research_findings: list[dict[str, Any]] = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        if verbose:
            print(f"  [iteration {iteration}] requesting next action from model...")

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            if verbose:
                print(f"  [done] agent finished after {iteration} iterations")
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results_for_model = []
        for block in tool_use_blocks:
            if verbose:
                print(f"  [tool] {block.name}({json.dumps(block.input)[:80]})")

            result = execute_tool(block.name, block.input)
            research_findings.append(
                {"tool": block.name, "input": block.input, "result": result}
            )
            tool_results_for_model.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)[:6000],  # cap to control context
                }
            )

        messages.append({"role": "user", "content": tool_results_for_model})

    if verbose:
        print(f"  [collected] {len(research_findings)} pieces of evidence")

    return research_findings
