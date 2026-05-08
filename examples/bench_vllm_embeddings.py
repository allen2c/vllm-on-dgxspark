"""
Benchmark a vLLM OpenAI-compatible embeddings API.

Default target:
  base_url = http://localhost:8944/v1
  model    = voyage-4-nano

Install:
  pip install openai numpy tqdm faker

Examples:
  python bench_vllm_embeddings.py

  python bench_vllm_embeddings.py \
    --num-texts 5000 \
    --batch-size 64 \
    --concurrency 16 \
    --repeat 4 \
    --random-input

  python bench_vllm_embeddings.py \
    --mode query \
    --num-texts 1000 \
    --batch-size 32 \
    --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from faker import Faker
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


QUERY_PREFIX = "Represent the query for retrieving supporting documents: "
DOC_PREFIX = "Represent the document for retrieval: "

InputMode = Literal["query", "document", "raw"]


@dataclass(frozen=True)
class RequestResult:
    success: bool
    latency_s: float
    num_inputs: int
    embedding_dim: int = 0
    error: str = ""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def apply_voyage_prefix(text: str, mode: InputMode) -> str:
    if mode == "query":
        return QUERY_PREFIX + text
    if mode == "document":
        return DOC_PREFIX + text
    return text


def make_static_text(index: int, repeat: int) -> str:
    base = (
        f"This is test document number {index}. "
        "It contains mixed English and Chinese text for embedding benchmark. "
        "台北牛肉麵、向量檢索、RAG、語意搜尋、Voyage embedding performance test. "
    )
    return base * repeat


def make_random_text(fake: Faker, repeat: int) -> str:
    """
    Generate randomized mixed English / Traditional Chinese benchmark text.

    A UUID is included to reduce repeated-content effects and make each input unique.
    """
    unique_id = str(uuid.uuid4())

    topics = [
        "RAG search",
        "embedding benchmark",
        "semantic retrieval",
        "hybrid search",
        "vector database",
        "Voyage embedding",
        "vLLM performance",
        "台北牛肉麵",
        "語意搜尋",
        "向量檢索",
        "文件相似度",
        "知識庫檢索",
        "多語言搜尋",
    ]

    parts: list[str] = []

    for _ in range(repeat):
        topic = random.choice(topics)
        parts.append(
            "\n".join(
                [
                    f"id={unique_id}",
                    f"topic={topic}",
                    f"name={fake.name()}",
                    f"company={fake.company()}",
                    f"city={fake.city()}",
                    f"address={fake.address()}",
                    f"sentence={fake.sentence(nb_words=random.randint(8, 24))}",
                    f"paragraph={fake.paragraph(nb_sentences=random.randint(2, 6))}",
                    f"zh_text=這是一段用於測試 embedding API 效能的隨機中文內容，主題是 {topic}。",
                ]
            )
        )

    return "\n\n".join(parts)


def generate_texts(
    *,
    num_texts: int,
    mode: InputMode,
    repeat: int,
    random_input: bool,
    seed: int | None,
) -> list[str]:
    if num_texts <= 0:
        raise ValueError("num_texts must be > 0")
    if repeat <= 0:
        raise ValueError("repeat must be > 0")

    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    fake = Faker(["en_US", "zh_TW"])

    texts: list[str] = []
    for i in range(num_texts):
        if random_input:
            text = make_random_text(fake=fake, repeat=repeat)
        else:
            text = make_static_text(index=i, repeat=repeat)

        texts.append(apply_voyage_prefix(text, mode))

    return texts


def approx_token_count(texts: Iterable[str]) -> float:
    """
    Very rough approximation.

    This is enough for a quick throughput sanity check, but not a replacement for
    model-specific tokenization.
    """
    total_chars = sum(len(text) for text in texts)
    return total_chars / 4.0


async def embed_batch(
    *,
    client: AsyncOpenAI,
    model: str,
    batch: list[str],
    timeout_s: float,
) -> RequestResult:
    start = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            client.embeddings.create(
                model=model,
                input=batch,
            ),
            timeout=timeout_s,
        )

        latency_s = time.perf_counter() - start
        embedding_dim = len(response.data[0].embedding) if response.data else 0

        return RequestResult(
            success=True,
            latency_s=latency_s,
            num_inputs=len(batch),
            embedding_dim=embedding_dim,
        )

    except Exception as exc:
        latency_s = time.perf_counter() - start
        return RequestResult(
            success=False,
            latency_s=latency_s,
            num_inputs=len(batch),
            error=repr(exc),
        )


async def run_warmup(
    *,
    client: AsyncOpenAI,
    model: str,
    request_batches: list[list[str]],
    warmup_requests: int,
    timeout_s: float,
) -> None:
    if warmup_requests <= 0:
        return

    warmup_batches = request_batches[:warmup_requests]

    print(f"Running warmup: {len(warmup_batches)} request(s)")
    for batch in warmup_batches:
        await embed_batch(
            client=client,
            model=model,
            batch=batch,
            timeout_s=timeout_s,
        )
    print("Warmup done.\n")


async def run_benchmark(args: argparse.Namespace) -> None:
    texts = generate_texts(
        num_texts=args.num_texts,
        mode=args.mode,
        repeat=args.repeat,
        random_input=args.random_input,
        seed=args.seed,
    )

    request_batches = batched(texts, args.batch_size)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    print_config(args=args, num_requests=len(request_batches))

    await run_warmup(
        client=client,
        model=args.model,
        request_batches=request_batches,
        warmup_requests=args.warmup,
        timeout_s=args.timeout,
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded_embed(batch: list[str]) -> RequestResult:
        async with semaphore:
            return await embed_batch(
                client=client,
                model=args.model,
                batch=batch,
                timeout_s=args.timeout,
            )

    benchmark_start = time.perf_counter()

    results: list[RequestResult] = await tqdm_asyncio.gather(
        *[bounded_embed(batch) for batch in request_batches],
        desc="Embedding requests",
    )

    total_time_s = time.perf_counter() - benchmark_start

    print_results(
        results=results,
        texts=texts,
        total_time_s=total_time_s,
    )


def print_config(*, args: argparse.Namespace, num_requests: int) -> None:
    print("Benchmark config")
    print("----------------")
    print(f"Base URL:       {args.base_url}")
    print(f"Model:          {args.model}")
    print(f"Total texts:    {args.num_texts}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Requests:       {num_requests}")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Text repeat:    {args.repeat}")
    print(f"Mode:           {args.mode}")
    print(f"Random input:   {args.random_input}")
    print(f"Seed:           {args.seed}")
    print()


def print_results(
    *,
    results: list[RequestResult],
    texts: list[str],
    total_time_s: float,
) -> None:
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    successful_inputs = sum(r.num_inputs for r in successes)
    failed_inputs = sum(r.num_inputs for r in failures)

    latencies = [r.latency_s for r in successes]
    embedding_dims = [r.embedding_dim for r in successes if r.embedding_dim > 0]
    embedding_dim = embedding_dims[0] if embedding_dims else 0

    successful_texts = texts[:successful_inputs]
    total_chars = sum(len(text) for text in successful_texts)
    approx_tokens = approx_token_count(successful_texts)

    requests_per_sec = len(successes) / total_time_s if total_time_s > 0 else 0.0
    inputs_per_sec = successful_inputs / total_time_s if total_time_s > 0 else 0.0
    chars_per_sec = total_chars / total_time_s if total_time_s > 0 else 0.0
    tokens_per_sec = approx_tokens / total_time_s if total_time_s > 0 else 0.0

    print()
    print("Benchmark results")
    print("-----------------")
    print(f"Total wall time:          {total_time_s:.3f} s")
    print(f"Successful requests:      {len(successes)}")
    print(f"Failed requests:          {len(failures)}")
    print(f"Successful inputs:        {successful_inputs}")
    print(f"Failed inputs:            {failed_inputs}")
    print(f"Embedding dimension:      {embedding_dim}")
    print(f"Requests/sec:             {requests_per_sec:.2f}")
    print(f"Inputs/sec:               {inputs_per_sec:.2f}")
    print(f"Approx chars/sec:         {chars_per_sec:.2f}")
    print(f"Approx tokens/sec:        {tokens_per_sec:.2f}")

    if latencies:
        print()
        print("Latency per request")
        print("-------------------")
        print(f"Mean:                     {statistics.mean(latencies):.4f} s")
        print(f"Median / p50:             {percentile(latencies, 50):.4f} s")
        print(f"p90:                      {percentile(latencies, 90):.4f} s")
        print(f"p95:                      {percentile(latencies, 95):.4f} s")
        print(f"p99:                      {percentile(latencies, 99):.4f} s")
        print(f"Min:                      {min(latencies):.4f} s")
        print(f"Max:                      {max(latencies):.4f} s")

    if failures:
        print()
        print("Sample errors")
        print("-------------")
        for failure in failures[:5]:
            print(failure.error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a vLLM OpenAI-compatible embeddings API."
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:8944/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key. vLLM usually accepts any value unless configured otherwise.",
    )
    parser.add_argument(
        "--model",
        default="voyage-4-nano",
        help="Served model name.",
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=1000,
        help="Total number of texts to embed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts per embeddings request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent requests.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=4,
        help="Repeat generated content this many times to simulate longer inputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "document", "raw"],
        default="document",
        help="Whether to add Voyage query/document prompt prefix.",
    )
    parser.add_argument(
        "--random-input",
        action="store_true",
        help="Use Faker/randomized input to reduce cache effects.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible Faker/random input.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup requests before the measured benchmark.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.num_texts <= 0:
        raise SystemExit("--num-texts must be > 0")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
