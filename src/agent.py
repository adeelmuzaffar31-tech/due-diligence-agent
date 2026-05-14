"""The agent orchestrator — the brain that decides what to research.

Uses tool-use / function-calling in an agentic loop: the model plans
research steps, calls tools, sees results, and continues until it has
enough evidence to stop. Returns all collected research for synthesis.

Works with both Claude (Anthropic) and GPT-4 (OpenAI) via the
llm_provider abstraction. Switch providers via LLM_PROVIDER env var.
"""

from __future__ import annotations

import json
from typing import Any

from .llm_provider import (
    append_assistant_message,
    append_tool_results,
    call_llm,
    get_model_name,
    get_provider,
)
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
    if verbose:
        print(f"  [provider] {get_provider()} ({get_model_name()})")

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

        response = call_llm(
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        append_assistant_message(messages, response)

        if response.stop_reason == "end_turn":
            if verbose:
                print(f"  [done] agent finished after {iteration} iterations")
            break

        if not response.tool_calls:
            break

        results: list[dict[str, Any]] = []
        for call in response.tool_calls:
            if verbose:
                print(
                    f"  [tool] {call.name}({json.dumps(call.arguments)[:80]})"
                )
            result = execute_tool(call.name, call.arguments)
            research_findings.append(
                {"tool": call.name, "input": call.arguments, "result": result}
            )
            results.append(result)

        append_tool_results(messages, response.tool_calls, results)

    if verbose:
        print(f"  [collected] {len(research_findings)} pieces of evidence")

    return research_findings
