#!/usr/bin/env bash
set -euo pipefail

# openai-rq consumer (worker) — runs on the in-cloud inference box.
# Pulls jobs off the Redis queue (XREADGROUP) and relays each one to the local
# vLLM OpenAI-compatible server, writing the result back to Redis. Outbound-only:
# it dials Redis and localhost vLLM; nothing connects *into* this box.
#
# OPENAI_RQ_REDIS_URL is loaded from .env by direnv. The backend credential, if
# any, is read from OPENAI_API_KEY by the CLI and is injected worker-side only
# (it never transits Redis).

: "${OPENAI_RQ_REDIS_URL:?set OPENAI_RQ_REDIS_URL (direnv loads it from .env)}"

BACKEND_URL="${OPENAI_RQ_BACKEND_URL:-http://localhost:8000/v1}"
CONCURRENCY="${OPENAI_RQ_CONCURRENCY:-16}"
GROUP="${OPENAI_RQ_GROUP:-openai-rq}"
STREAM_FLUSH_MS="${OPENAI_RQ_STREAM_FLUSH_MS:-50}"
RESULT_TTL_S="${OPENAI_RQ_RESULT_TTL_S:-600}"
MAX_RETRIES="${OPENAI_RQ_MAX_RETRIES:-3}"

exec openai-rq worker \
	--redis-url "$OPENAI_RQ_REDIS_URL" \
	--openai-base-url "$BACKEND_URL" \
	--concurrency "$CONCURRENCY" \
	--group "$GROUP" \
	--stream-flush-ms "$STREAM_FLUSH_MS" \
	--result-ttl-s "$RESULT_TTL_S" \
	--max-retries "$MAX_RETRIES"
