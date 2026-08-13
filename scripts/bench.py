#!/usr/bin/env python3
"""Concurrency benchmark for the DeepSeek-V4-Flash-0731 serve.
Reports per-stream and aggregate decode tok/s at each concurrency level.
Usage: BASE=http://HEAD_IP:8000/v1 python3 bench.py 1 2 3 6
Requires: aiohttp  (uv run --with aiohttp python3 bench.py ...)
"""
import asyncio, time, sys, os
BASE = os.environ.get("BASE", "http://192.168.0.211:8000/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
# a coding prompt -> high DSpark acceptance; swap to see content-driven spread
PROMPT = os.environ.get("PROMPT",
    "Write a Python binary search tree with insert, delete, and inorder traversal; explain each method.")
MAXTOK = int(os.environ.get("MAXTOK", "256"))

async def one(sem, res):
    import aiohttp
    async with sem, aiohttp.ClientSession() as s:
        body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAXTOK, "temperature": 0.7}
        t0 = time.time()
        async with s.post(f"{BASE}/chat/completions", json=body) as r:
            d = await r.json()
        res.append((d.get("usage", {}).get("completion_tokens", 0), time.time() - t0))

async def sweep(c):
    sem = asyncio.Semaphore(c); res = []; t0 = time.time()
    await asyncio.gather(*[one(sem, res) for _ in range(c)])
    wall = time.time() - t0; toks = sum(x for x, _ in res)
    print(f"c{c:<3} per-stream={toks/wall/c:6.1f} tok/s  agg={toks/wall:7.1f}  wall={wall:5.1f}s  toks={toks}")

async def main():
    levels = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 6]
    await sweep(1)  # warm
    for c in levels:
        await sweep(c)

asyncio.run(main())
