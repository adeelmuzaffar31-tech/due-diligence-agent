"""LLM provider abstraction — supports both Anthropic (Claude) and OpenAI (GPT).

The agent is model-agnostic. You can switch providers via the LLM_PROVIDER
environment variable. The same agent loop works with either model's tool-use
(Claude) or function-calling (GPT) API. This module hides the differences.

Usage:
    LLM_PROVIDER=anthropic  → uses Claude (default)
    LLM_PROVIDER=openai     → uses any GPT or O-series model

This module is designed to be forward-compatible: when OpenAI launches a new
model with a different parameter signature, the OpenAI call automatically
retries with the corrected parameter name based on the API's error response.
"""

from __future__ import annotations

import json
import os
import time
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


# ============================================================
# OpenAI model capability detection
# ============================================================
#
# OpenAI's models have diverged over time:
#   - GPT-4 family:        uses max_tokens, supports temperature/top_p
#   - GPT-5 family:        uses max_completion_tokens, fixed temperature
#   - O-series (reasoning) uses max_completion_tokens, no temperature, no tools (older o1)
#
# Rather than maintaining a list that goes stale, we detect capabilities
# from the model name prefix AND fall back gracefully on API errors.

def _is_reasoning_model(model_name: str) -> bool:
    """O-series reasoning models have stricter parameter constraints."""
    name = model_name.lower()
    return any(name.startswith(prefix) for prefix in ("o1", "o3", "o4"))


def _uses_max_completion_tokens(model_name: str) -> bool:
    """Newer model families use max_completion_tokens instead of max_tokens."""
    name = model_name.lower()
    return any(
        name.startswith(prefix)
        for prefix in ("gpt-5", "o1", "o3", "o4", "chatgpt-5")
    )


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
# OpenAI (GPT) implementation — universal compatibility
# ============================================================

def _build_openai_kwargs(
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    """Build OpenAI request kwargs based on detected model capabilities."""
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }

    # Token limit parameter naming changed in GPT-5 / O-series
    if _uses_max_completion_tokens(model_name):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    # Tools — older O1 models didn't support tools, but newer ones do.
    # Pass them; if rejected, we'll retry without them.
    if tools:
        kwargs["tools"] = _convert_tools_for_openai(tools)

    return kwargs


def _call_openai_with_retry(client: Any, kwargs: dict[str, Any]) -> Any:
    """Call OpenAI and auto-recover from common parameter incompatibilities.

    OpenAI's API surface changes between model families. Rather than
    maintaining a perfect compatibility matrix, we react to the API's own
    error messages and adapt.
    """
    max_retries = 4
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            error_msg = str(exc).lower()

            # Recoverable: parameter renamed
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                if "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue

            # Recoverable: opposite — model wants the old name
            if "max_completion_tokens" in error_msg and "not supported" in error_msg:
                if "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    continue

            # Recoverable: temperature unsupported (some O-series, GPT-5)
            if "temperature" in error_msg and "unsupported" in error_msg:
                kwargs.pop("temperature", None)
                continue

            # Recoverable: tools/functions unsupported on this model
            if any(t in error_msg for t in ("tools", "function")) and "unsupported" in error_msg:
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                continue

            # Recoverable: rate limit — back off and retry
            if "rate" in error_msg and "limit" in error_msg:
                time.sleep(2 ** attempt)
                continue

            # Unrecoverable — raise with a clean message
            raise

    raise RuntimeError(
        f"OpenAI call failed after {max_retries} adaptive retries. "
        f"Last kwargs: {list(kwargs.keys())}"
    )


def _call_openai(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 4096,
) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = get_model_name()

    # OpenAI uses a flat messages array with system as first message.
    # Some reasoning models (older o1) reject the "system" role entirely
    # and require it merged into the first user message.
    if _is_reasoning_model(model_name) and model_name.startswith("o1"):
        # Older o1 models: merge system into first user message
        merged_messages = list(messages)
        if merged_messages and merged_messages[0].get("role") == "user":
            existing = merged_messages[0].get("content", "")
            if isinstance(existing, str):
                merged_messages[0] = {
                    "role": "user",
                    "content": f"{system}\n\n{existing}",
                }
            else:
                merged_messages.insert(0, {"role": "user", "content": system})
        openai_messages = merged_messages
    else:
        openai_messages = [{"role": "system", "content": system}] + messages

    kwargs = _build_openai_kwargs(model_name, openai_messages, tools, max_tokens)
    response = _call_openai_with_retry(client, kwargs)

    choice = response.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Malformed args — pass empty dict and let the tool reject
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

    # Build clean assistant message dict for history. OpenAI requires
    # tool_calls in a specific format on subsequent calls.
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if getattr(message, "tool_calls", None):
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

    stop = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(
        text=message.content or "",
        tool_calls=tool_calls,
        raw_message=assistant_msg,
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
    if response is None or response.raw_message is None:
        return
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