#!/usr/bin/env python3
"""Verify a vLLM OpenAI-compatible endpoint: basic completion + tool calling.

Works against either target:
  - Phase 1 pod (local):   BASE_URL=http://localhost:8000/v1            API_KEY=EMPTY
  - Phase 2 serverless:    BASE_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
                           API_KEY=<RUNPOD_API_KEY>

Usage:
  pip install openai
  BASE_URL=... API_KEY=... MODEL_ID=... python scripts/test_endpoint.py
"""
import os
import sys

from openai import OpenAI

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")
MODEL_ID = os.environ.get("MODEL_ID")
if not MODEL_ID:
    sys.exit("Set MODEL_ID to the served model name (HF repo id or local path).")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def test_basic_completion() -> None:
    print("== Test 1: basic chat completion ==")
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Write a Python function that reverses a string."}],
        max_tokens=256,
    )
    content = resp.choices[0].message.content or ""
    print(content[:500])
    assert content.strip(), "empty completion"
    print("OK\n")


def test_tool_calling() -> None:
    print("== Test 2: tool calling (validates --tool-call-parser) ==")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }
    ]
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "What's the weather in Berlin right now?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=256,
    )
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    print("tool_calls:", calls)
    assert calls, "model did not emit a tool call — check the tool-call parser for this model"
    assert calls[0].function.name == "get_weather", f"unexpected tool: {calls[0].function.name}"
    print("OK\n")


if __name__ == "__main__":
    print(f"endpoint={BASE_URL} model={MODEL_ID}\n")
    test_basic_completion()
    test_tool_calling()
    print("All checks passed.")
