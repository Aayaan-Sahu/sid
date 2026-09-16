"""Live view of what each server is doing. Run it in a second terminal while the demo runs.

    python demo/watch.py                                        # both demo servers
    python demo/watch.py http://localhost:8000                  # just one

Columns:
    running / waiting   requests being decoded right now / queued for their turn
    prefill, decode     tokens per second through each path
    verified            tokens per second confirmed by the verifier (deterministic server only)
    rollbacks           drafted tokens the verifier threw away and recomputed; this is the
                        load-dependent arithmetic being caught before a client ever sees it
    kv cache            share of the kv cache in use
"""
import argparse
import asyncio
import sys
import time

import httpx

BOLD, GREEN, RED, DIM, OFF = "\033[1m", "\033[32m", "\033[31m", "\033[2m", "\033[0m"


async def poll(client, url) -> dict | None:
    try:
        return (await client.get(f"{url}/metrics", timeout=2.0)).json()
    except Exception:
        return None


def rate(now: dict, before: dict, key: str, seconds: float) -> float:
    if not before or seconds <= 0:
        return 0.0
    return max(now.get(key, 0) - before.get(key, 0), 0) / seconds


def render(rows: list[str], first: bool):
    if not first:
        sys.stdout.write(f"\033[{len(rows)}A")
    for row in rows:
        sys.stdout.write("\033[2K" + row + "\n")
    sys.stdout.flush()


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", default=["http://localhost:8000", "http://localhost:8001"])
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    urls = [u.rstrip("/").removesuffix("/v1") for u in args.urls]
    names = {}
    width = max(len(u) for u in urls)

    async with httpx.AsyncClient() as client:
        for url in urls:    # label each server by whether its verifier is on
            try:
                info = (await client.get(f"{url}/v1/sid/info", timeout=2.0)).json()
                names[url] = "deterministic" if info.get("enable_determinism") else "non-deterministic"
            except Exception:
                names[url] = url
        width = max(len(n) for n in names.values())

        before: dict[str, dict] = {}
        last = time.perf_counter()
        first = True
        while True:
            await asyncio.sleep(args.interval)
            now = time.perf_counter()
            seconds, last = now - last, now
            snapshots = await asyncio.gather(*[poll(client, url) for url in urls])
            rows = [f"  {BOLD}{'server':<{width}}  {'running':>8}{'waiting':>9}{'prefill/s':>11}{'decode/s':>10}"
                    f"{'verified/s':>12}{'rollbacks':>11}{'kv cache':>10}{OFF}"]
            for url, metrics in zip(urls, snapshots):
                name = names[url]
                if metrics is None:
                    rows.append(f"  {name:<{width}}  {DIM}unreachable{OFF}")
                    continue
                total_blocks = metrics.get("total_kvcache_blocks", 0) or 1
                used = 100 * (1 - metrics.get("free_kvcache_blocks", 0) / total_blocks)
                verified = rate(metrics, before.get(url), "verify_windows", seconds) * 32
                rollbacks = metrics.get("rollbacks", 0)
                rollback_cell = f"{RED}{rollbacks:>11}{OFF}" if rollbacks else f"{DIM}{'-':>11}{OFF}"
                rows.append(
                    f"  {name:<{width}}  {metrics.get('running', 0):>8}{metrics.get('waiting', 0):>9}"
                    f"{rate(metrics, before.get(url), 'prefill_tokens', seconds):>11,.0f}"
                    f"{rate(metrics, before.get(url), 'decode_tokens', seconds):>10,.0f}"
                    f"{verified:>12,.0f}" + rollback_cell + f"{used:>9.0f}%"
                )
                before[url] = metrics
            rows.append(f"  {DIM}ctrl-c to stop{OFF}")
            render(rows, first)
            first = False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
