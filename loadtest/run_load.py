"""Open-loop load generator for the ranking endpoint.

Why open loop
-------------
The obvious way to load-test is a pool of workers that each send a request, wait for the
reply, and send the next one. That is a *closed* loop, and it systematically hides tail
latency: when the server slows down, the client slows down with it and simply stops
offering load. The requests that would have arrived during the stall are never sent, so
they never appear in the percentiles. This is coordinated omission, and it is why closed-
loop numbers look good precisely when a service is behaving badly.

This generator instead schedules arrivals on a Poisson process at a fixed target rate,
independent of how the server is doing. If a response takes 200 ms, the arrivals that
were due in that window still happen, they queue, and their latency is measured from the
moment they were *due* - which is what the user on the landing page actually experiences.

Reported numbers
----------------
  * client latency  - due time to response received, including queueing
  * server latency  - the X-Process-Time-Ms header, handler time only
    The gap between them is queueing, and it is the first thing to look at when p99
    diverges from p50.
  * achieved rate vs target rate. If achieved is below target, the box is saturated and
    the percentiles describe an overloaded system, not a healthy one - say so rather
    than quoting them as the service's latency.

Percentiles are nearest-rank on the sorted sample (no interpolation), so every reported
value is a real observation.

Caveat worth stating in any report produced with this: on a small box the generator
competes with the server for CPU. Run `--rps` sweeps rather than a single point, and
treat the rate at which achieved falls behind target as the capacity estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class Sample:
    due: float
    sent: float
    finished: float
    status: int
    server_ms: float | None

    @property
    def client_ms(self) -> float:
        """Measured from when the request was *due*, so queueing is included."""
        return (self.finished - self.due) * 1000

    @property
    def service_ms(self) -> float:
        return (self.finished - self.sent) * 1000


@dataclass
class Result:
    target_rps: float
    duration_s: float
    samples: list[Sample] = field(default_factory=list)
    errors: int = 0
    max_in_flight: int = 0

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.status == 200]
        client = sorted(s.client_ms for s in ok)
        server = sorted(s.server_ms for s in ok if s.server_ms is not None)
        span = (max(s.finished for s in ok) - min(s.due for s in ok)) if ok else 0.0
        return {
            "target_rps": self.target_rps,
            "achieved_rps": round(len(ok) / span, 1) if span > 0 else 0.0,
            "requests": len(self.samples),
            "ok": len(ok),
            "errors": self.errors + sum(1 for s in self.samples if s.status != 200),
            "max_in_flight": self.max_in_flight,
            "client_ms": _percentiles(client),
            "server_ms": _percentiles(server),
        }


def _percentiles(values: list[float]) -> dict[str, float]:
    """Nearest-rank percentiles: every number returned is an observed value."""
    if not values:
        return {}
    def at(q: float) -> float:
        rank = max(1, min(len(values), int(-(-q * len(values) // 1))))
        return round(values[rank - 1], 2)
    return {
        "p50": at(0.50), "p90": at(0.90), "p95": at(0.95),
        "p99": at(0.99), "p999": at(0.999),
        "max": round(values[-1], 2), "mean": round(statistics.fmean(values), 2),
    }


async def _fire(client: httpx.AsyncClient, url: str, payload: bytes,
                due: float, result: Result, in_flight: list[int]) -> None:
    in_flight[0] += 1
    result.max_in_flight = max(result.max_in_flight, in_flight[0])
    sent = time.perf_counter()
    try:
        response = await client.post(url, content=payload,
                                     headers={"content-type": "application/json"})
        header = response.headers.get("X-Process-Time-Ms")
        result.samples.append(Sample(
            due=due, sent=sent, finished=time.perf_counter(),
            status=response.status_code,
            server_ms=float(header) if header else None,
        ))
    except Exception:  # noqa: BLE001 - a dropped connection is a load-test result
        result.errors += 1
    finally:
        in_flight[0] -= 1


async def run(url: str, payloads: list[bytes], target_rps: float, duration_s: float,
              warmup_s: float, connections: int) -> Result:
    result = Result(target_rps=target_rps, duration_s=duration_s)
    limits = httpx.Limits(max_connections=connections,
                          max_keepalive_connections=connections)
    rng = random.Random(1234)
    in_flight = [0]

    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        # Warm-up traffic is sent but not measured: the first requests to a fresh
        # worker pay allocation costs that are not representative of steady state.
        warm = Result(target_rps=target_rps, duration_s=warmup_s)
        await _drive(client, url, payloads, target_rps, warmup_s, warm, rng, in_flight)
        await _drive(client, url, payloads, target_rps, duration_s, result, rng, in_flight)
    return result


async def _drive(client: httpx.AsyncClient, url: str, payloads: list[bytes],
                 target_rps: float, duration_s: float, result: Result,
                 rng: random.Random, in_flight: list[int]) -> None:
    """Schedule Poisson arrivals and fire them regardless of server progress."""
    if duration_s <= 0:
        return
    tasks: list[asyncio.Task] = []
    start = time.perf_counter()
    due = start
    interval = 1.0 / target_rps

    while True:
        # Exponential gaps give a Poisson process: bursty like real traffic, rather
        # than the artificially smooth stream a fixed interval produces.
        due += rng.expovariate(1.0 / interval)
        if due - start > duration_s:
            break
        sleep_for = due - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        payload = payloads[len(tasks) % len(payloads)]
        tasks.append(asyncio.create_task(
            _fire(client, url, payload, due, result, in_flight)
        ))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def load_payloads(path: Path | None, count: int) -> list[bytes]:
    """Distinct users, so the run does not measure one perfectly cached code path."""
    from bl_ranking.serving.ranker import WARMUP_USER

    if path and path.exists():
        records = json.loads(path.read_text())
        return [json.dumps(r).encode() for r in records]

    rng = random.Random(7)
    cities = ["Fort Lauderdale", "Austin", "Chicago", "Phoenix", "Seattle", "Miami"]
    names = ["Rigoberto", "Michael", "Jennifer", "Svetlana", "Ahmed", "Priya", "Marcus"]
    credits = ["Very Poor - Under 550", "Poor - 550 to 599", "Fair - 600 to 649",
               "Good - 650 to 719", "Excellent - 720 and above"]
    revenues = ["$5,000 - $9,999", "$20,000 - $49,999", "$50,000 - $99,999", "$200,000+"]
    amounts = ["$10,000 - $24,999", "$25,000 - $49,999", "$100,000 - $199,999"]

    payloads = []
    for i in range(count):
        user = dict(WARMUP_USER)
        user["auto_city"] = rng.choice(cities)
        user["fname"] = rng.choice(names)
        user["credit_score"] = rng.choice(credits)
        user["monthly_revenue"] = rng.choice(revenues)
        user["loan_amount"] = rng.choice(amounts)
        user["session_dt"] = f"2026-01-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:15:00"
        user["register_date"] = f"2026-01-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:17:30"
        user["cellphone"] = 2010000000 + i
        payloads.append(json.dumps(user).encode())
    return payloads


def render(summaries: list[dict]) -> str:
    lines = [
        "",
        f"{'target':>8} {'achieved':>9} {'ok':>7} {'err':>5} "
        f"{'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'p99.9':>8} {'max':>9}",
        "-" * 92,
    ]
    for s in summaries:
        c = s["client_ms"]
        if not c:
            lines.append(f"{s['target_rps']:>8.0f} {'-':>9} {s['ok']:>7} {s['errors']:>5}")
            continue
        lines.append(
            f"{s['target_rps']:>8.0f} {s['achieved_rps']:>9.1f} {s['ok']:>7} "
            f"{s['errors']:>5} {c['p50']:>8.2f} {c['p90']:>8.2f} {c['p95']:>8.2f} "
            f"{c['p99']:>8.2f} {c['p999']:>8.2f} {c['max']:>9.2f}"
        )
    lines.append("")
    lines.append("client latency in ms, measured from scheduled arrival (queueing included)")
    for s in summaries:
        if s["server_ms"]:
            lines.append(
                f"  at {s['target_rps']:>5.0f} rps  server-side handler: "
                f"p50 {s['server_ms']['p50']:.2f}  p99 {s['server_ms']['p99']:.2f} ms"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080/rank")
    parser.add_argument("--rps", default="50,100,200,400",
                        help="comma-separated target rates to sweep")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--connections", type=int, default=64)
    parser.add_argument("--payloads", type=Path, default=None)
    parser.add_argument("--payload-count", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payloads = load_payloads(args.payloads, args.payload_count)
    summaries = []
    for rps in [float(x) for x in args.rps.split(",")]:
        result = asyncio.run(run(args.url, payloads, rps, args.duration,
                                 args.warmup, args.connections))
        summaries.append(result.summary())
        print(f"  {rps:>6.0f} rps done: {summaries[-1]['ok']} ok, "
              f"{summaries[-1]['errors']} errors", flush=True)

    report = render(summaries)
    print(report)
    if args.out:
        # JSON and the human report go to separate files: mixing them makes the JSON
        # unparseable by anything downstream, which is the whole reason to emit it.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2) + "\n")
        args.out.with_suffix(".txt").write_text(report + "\n")
        print(f"\nwritten to {args.out} and {args.out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
