"""LLM provider abstraction — supports both Anthropic (Claude) and OpenAI (GPT).

The agent is model-agnostic. You can switch providers via the LLM_PROVIDER
environment variable. The same agent loop works with either model's tool-use
(Claude) or function-calling (GPT) API. This module hides the differences.

Usage:
    LLM_PROVIDER=anthropic  → uses Claude (default)
    LLM_PROVIDER=openai     → uses GPT-4
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal


Provider = Literal["anthropic", "openai"]


@dataclass
class ToolCall:
    """Normalised tool-call request from any provider."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalised response from any provider."""
    text: str
    tool_calls: list[ToolCall]
    raw_message: Any  # original provider-specific message for history
    stop_reason: str  # "end_turn" or "tool_use" (normalised)


def get_provider() -> Provider:
    """Read provider from env. Defaults to Anthropic."""
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()  # type: ignore[return-value]


def get_model_name() -> str:
    """Return the configured model name for the active provider."""
    provider = get_provider()
    if provider == "anthropic":
        return os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o")
    raise ValueError(f"Unknown provider: {provider}")


def _convert_tools_for_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style tool defs to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# ============================================================
# Anthropic (Claude) implementation
# ============================================================

def _call_anthropic(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 4096,
) -> LLMResponse:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=get_model_name(),
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        messages=messages,
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            )

    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        raw_message={"role": "assistant", "content": response.content},
        stop_reason="end_turn" if response.stop_reason == "end_turn" else "tool_use",
    )


def _format_anthropic_tool_results(
    tool_calls: list[ToolCall], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Format tool results as a user message for Anthropic."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result)[:6000],
            }
            for call, result in zip(tool_calls, results)
        ],
    }


# ============================================================
# OpenAI (GPT) implementation
# ============================================================

def _call_openai(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 4096,
) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # OpenAI uses a flat messages array with system as first message
    openai_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=get_model_name(),
        max_tokens=max_tokens,
        tools=_convert_tools_for_openai(tools),
        messages=openai_messages,
    )

    choice = response.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )

    stop = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(
        text=message.content or "",
        tool_calls=tool_calls,
        raw_message=message.model_dump(),
        stop_reason=stop,
    )


def _format_openai_tool_results(
    tool_calls: list[ToolCall], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Format tool results as separate tool messages for OpenAI."""
    return [
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)[:6000],
        }
        for call, result in zip(tool_calls, results)
    ]


# ============================================================
# Public API — provider-agnostic
# ============================================================

def call_llm(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Call the configured LLM with optional tools. Provider-agnostic."""
    provider = get_provider()
    if provider == "anthropic":
        return _call_anthropic(system, messages, tools or [], max_tokens)
    if provider == "openai":
        return _call_openai(system, messages, tools or [], max_tokens)
    raise ValueError(f"Unknown provider: {provider}")


def append_assistant_message(
    messages: list[dict[str, Any]], response: LLMResponse
) -> None:
    """Append the assistant's message to history in the provider's format."""
    if get_provider() == "anthropic":
        messages.append(response.raw_message)
    else:
        # OpenAI expects a dict with role/content/tool_calls
        messages.append(response.raw_message)


def append_tool_results(
    messages: list[dict[str, Any]],
    tool_calls: list[ToolCall],
    results: list[dict[str, Any]],
) -> None:
    """Append tool results to history in the provider's format."""
    if get_provider() == "anthropic":
        messages.append(_format_anthropic_tool_results(tool_calls, results))
    else:
        messages.extend(_format_openai_tool_results(tool_calls, results))
