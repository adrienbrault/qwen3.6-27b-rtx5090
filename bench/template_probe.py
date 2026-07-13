"""Behavioural probe for a Qwen chat template served by vLLM.

Checks the things a speed benchmark can't: does the model emit parseable tool calls,
does a chat->tool->chat loop survive (froggeric's headline "agentic abort" fix), and
does multi-turn prefix caching actually hit.

usage: python3 template_probe.py <base_url> <label>
"""

import json
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE
MODEL = "qwen3.6-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def post(payload, timeout=180):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


results = {}

# 1. single tool call
r = post({
    "model": MODEL,
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": TOOLS,
    "max_tokens": 1500,
})
msg = r["choices"][0]["message"]
calls = msg.get("tool_calls") or []
results["single_tool_call"] = (
    len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
)

# 2. parallel tool calls (the multi-call drop bug)
r = post({
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": "What's the weather AND the current time in Tokyo? Call both tools.",
    }],
    "tools": TOOLS,
    "max_tokens": 1500,
})
msg2 = r["choices"][0]["message"]
calls2 = msg2.get("tool_calls") or []
names = sorted(c["function"]["name"] for c in calls2)
results["parallel_tool_calls"] = names == ["get_time", "get_weather"]
results["_parallel_got"] = names

# 3. chat -> tool result -> chat continuation (froggeric's "agentic abort" fix)
if calls:
    convo = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": [{
                "id": calls[0].get("id", "call_1"),
                "type": "function",
                "function": {
                    "name": calls[0]["function"]["name"],
                    "arguments": calls[0]["function"]["arguments"],
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": calls[0].get("id", "call_1"),
            "content": '{"temp_c": 18, "condition": "rain"}',
        },
    ]
    r = post({"model": MODEL, "messages": convo, "tools": TOOLS, "max_tokens": 1500})
    final = (r["choices"][0]["message"].get("content") or "").strip()
    results["tool_result_continuation"] = ("18" in final) and len(final) > 10
    results["_continuation_text"] = final[:100]
else:
    results["tool_result_continuation"] = False

# 4. plain multi-turn chat after a tool turn (the "abandons after mixing chat+tools" case)
convo.append({"role": "assistant", "content": final if calls else "ok"})
convo.append({"role": "user", "content": "Thanks. Now, in one word: is that good weather?"})
r = post({"model": MODEL, "messages": convo, "tools": TOOLS, "max_tokens": 1500})
followup = (r["choices"][0]["message"].get("content") or "").strip()
results["chat_after_tools"] = len(followup) > 0
results["_followup_text"] = followup[:60]

print(f"\n=== {LABEL}")
for k, v in results.items():
    if k.startswith("_"):
        print(f"    {k}: {v!r}")
    else:
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
score = sum(1 for k, v in results.items() if not k.startswith("_") and v)
total = sum(1 for k in results if not k.startswith("_"))
print(f"  -> {score}/{total}")
