"""Measure prefix-cache hit rate over a growing multi-turn conversation.

froggeric's template claims to enforce chronological history for near-100% KV cache hits.
This is the one claim a behavioural tool-probe can't see. Drive a realistic agentic
pattern (same long system prompt + growing turn history) and read vLLM's own counter.

usage: python3 prefix_probe.py <base_url> <label>
"""

import json
import re
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE
MODEL = "qwen3.6-27b"

SYSTEM = (
    "You are a coding assistant working in a large repository. "
    "Follow the house style. Be concise. " + ("Context filler. " * 400)
)


def post(messages):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps({"model": MODEL, "messages": messages, "max_tokens": 300}).encode(),
        {"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=180))
    return r["choices"][0]["message"].get("content") or "", r["usage"]


def metrics():
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=30) as f:
        body = f.read().decode()
    q = re.search(r"^vllm:prefix_cache_queries_total\S*\s+([\d.e+]+)$", body, re.M)
    h = re.search(r"^vllm:prefix_cache_hits_total\S*\s+([\d.e+]+)$", body, re.M)
    return (float(q.group(1)) if q else 0.0, float(h.group(1)) if h else 0.0)


q0, h0 = metrics()

convo = [{"role": "system", "content": SYSTEM}]
questions = [
    "What is 2+2? One word.",
    "And 3+3? One word.",
    "And 4+4? One word.",
    "And 5+5? One word.",
    "And 6+6? One word.",
]
for question in questions:
    convo.append({"role": "user", "content": question})
    reply, usage = post(convo)
    convo.append({"role": "assistant", "content": reply})

q1, h1 = metrics()
dq, dh = q1 - q0, h1 - h0
rate = (dh / dq * 100) if dq else 0.0
print(f"=== {LABEL}")
print(f"  prefix cache: {dh:.0f}/{dq:.0f} blocks hit -> {rate:.1f}%")
