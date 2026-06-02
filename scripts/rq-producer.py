#!/usr/bin/env python
"""openai-rq producer (client demo) — runs on the network-restricted client side.

Sends a chat completion through the Redis queue using OpenAIRQ, a drop-in
openai.OpenAI subclass: the request rides Redis to the in-cloud worker and the
result comes back the same way. The only outbound dependency is the Redis
endpoint in OPENAI_RQ_REDIS_URL (loaded from .env by direnv).

Usage:
    scripts/rq-producer.py "your prompt here"
    scripts/rq-producer.py --stream "count to three"
    echo "prompt from stdin" | scripts/rq-producer.py
    scripts/rq-producer.py --model ... --max-tokens 512 "..."
"""

from __future__ import annotations

import argparse
import os
import sys

from openai_rq import OpenAIRQ

DEFAULT_MODEL = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_PROMPT = "Say hello in one short sentence."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send a chat completion through the openai-rq Redis queue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "prompt",
        nargs="?",
        help="prompt text; falls back to stdin, then a default greeting",
    )
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    p.add_argument("--stream", action="store_true", help="stream the response")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--redis-url",
        default=None,
        help="defaults to $OPENAI_RQ_REDIS_URL (kept out of --help to avoid "
        "leaking the secret)",
    )
    return p.parse_args()


def resolve_prompt(arg: str | None) -> str:
    if arg:
        return arg
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return DEFAULT_PROMPT


def main() -> int:
    args = parse_args()
    redis_url = args.redis_url or os.environ.get("OPENAI_RQ_REDIS_URL")
    if not redis_url:
        sys.exit("error: set OPENAI_RQ_REDIS_URL (direnv loads it from .env)")

    client = OpenAIRQ(redis_url=redis_url)
    messages = [{"role": "user", "content": resolve_prompt(args.prompt)}]

    if args.stream:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            stream=True,
        )
        for chunk in resp:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        print()
    else:
        resp = client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=args.max_tokens
        )
        print(resp.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
