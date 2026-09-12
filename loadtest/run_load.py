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

Which client is slow
--------------------
Open loop has a failure mode of its own, and this harness walked straight into it. One
asyncio process on a four-core box, three uvicorn workers on the same four cores: past a
couple of hundred requests per second the *generator* is what runs out of CPU. Its own
scheduling loop falls behind, requests go out long after they were due, and because client
latency is measured from the due time, every second of the generator's lateness is charged
to the server. The numbers that produced were not merely noisy, they were inverted: the
harness reported 54 s p50 at 400 rps while an independent client, asking the same server at
the same moment, got 8.3 ms.

So the send lag - how late each request actually went out - is measured and reported, and a
run whose p99 send lag exceeds `--max-send-lag-ms` is marked `generator_saturated`. Those
numbers describe this program, not the service, and the report says so instead of leaving
it to be inferred from `achieved` being under target.

`--processes N` is the other half. Arrivals are split across N independent generator
processes, so the offered rate can exceed what one event loop can schedule. Three
processes on this box offer 300 rps at a p99 send lag of a few milliseconds, where one
process cannot.

Reported numbers
----------------
  * send lag       - due time to the moment the request was actually sent. The validity
                     check: if this is not small, nothing else on the row means anything.
  * service        - sent to response received. What the server plus the network did.
  * client latency - due time to response received, so service plus send lag plus
                     queueing. What a user would experience *if* the generator kept up.
  * server latency - the X-Process-Time-Ms header. Handler time as the worker measured it,
                     which includes any wait inside that worker's own event loop - so the
                     gap between it and `service` is connection and OS queueing, not "all
                     of the queueing".
  * offered / ok / achieved. achieved is ok over the *offered* window, so it cannot read
    above target because the drain ran long.

Percentiles are nearest-rank on the sorted sample (no interpolation), so every reported
value is a real observation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
        """Measured from when the request was *due*, so queueing is included.

        Only meaningful while send_lag_ms is small. Otherwise this is mostly a measure of
        how far behind the generator's own scheduling loop had fallen.
        """
        return (self.finished - self.due) * 1000

    @property
    def service_ms(self) -> float:
        """Measured from when the request actually went out. The server's side of it."""
        return (self.finished - self.sent) * 1000

    @property
    def send_lag_ms(self) -> float:
        """How late this request went out. The validity check for everything else."""
        return (self.sent - self.due) * 1000


@dataclass
class Result:
    target_rps: float
    duration_s: float
    samples: list[Sample] = field(default_factory=list)
    errors: int = 0
    max_in_flight: int = 0
    offered: int = 0
    processes: int = 1
    max_send_lag_ms: float = 100.0
    connections: int = 128
    loadavg_1m: float = 0.0
    # Wall seconds from the start of the measured window to the last response, summed as a
    # max across generator processes. Not derived from the sample timestamps: perf_counter
    # has a different origin in every process, so differences within one are meaningful and
    # comparisons across them are not.
    span_s: float = 0.0

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.status == 200]
        client = sorted(s.client_ms for s in ok)
        service = sorted(s.service_ms for s in ok)
        server = sorted(s.server_ms for s in ok if s.server_ms is not None)
        # Every request that was scheduled, not only the ones that came back: a send lag
        # is a property of the attempt.
        lag = sorted(s.send_lag_ms for s in self.samples)
        lag_p = _percentiles(lag)

        # Throughput over the whole measurement, drain included. Dividing by the offered
        # window instead credits a server with every response that eventually arrived: a
        # run that offered 300 rps for 20 s and took 74 s to answer is not doing 297 rps.
        # Taken from the window start rather than from min(due) so a failed first request
        # cannot shrink the denominator.
        span = self.span_s or self.duration_s
        achieved = round(len(ok) / span, 1) if span > 0 else 0.0

        # Two ways this program, rather than the service, becomes the bottleneck.
        #
        #   late     - the scheduling loop fell behind, so requests went out after they
        #              were due and their client latency is mostly our own lateness.
        #   queued   - the loop kept up but far more requests are in flight than there are
        #              connections to carry them, so they sit in httpx's pool. That wait is
        #              inside `service_ms`, which makes the *server* look slow: 17.8 s p50
        #              from one generator at 300 rps, against 20 ms from three, same server.
        reasons = []
        if lag_p and lag_p["p99"] > self.max_send_lag_ms:
            reasons.append("late")
        # Four times the wire capacity, summed across processes: a brief burst past the
        # connection count is ordinary, a sustained backlog several times larger is the
        # client holding requests it cannot send.
        if self.max_in_flight > 4 * self.connections * self.processes:
            reasons.append("queued")
        saturated = bool(reasons)
        return {
            "target_rps": self.target_rps,
            "achieved_rps": achieved,
            "offered": self.offered,
            "requests": len(self.samples),
            "ok": len(ok),
            "errors": self.errors + sum(1 for s in self.samples if s.status != 200),
            "max_in_flight": self.max_in_flight,
            "processes": self.processes,
            # True means the row describes this program rather than the service. Reported
            # rather than inferred, because "achieved is under target" reads like a server
            # limit and is exactly what a saturated generator also produces.
            "generator_saturated": saturated,
            "saturation_reasons": reasons,
            "connections": self.connections,
            "span_s": round(span, 2),
            # What else the box was doing. A latency report from a machine under unrelated
            # load is not wrong, it is unattributable - and this harness shares four cores
            # with the server it measures.
            "loadavg_1m": self.loadavg_1m,
            "max_send_lag_ms": self.max_send_lag_ms,
            "send_lag_ms": lag_p,
            "service_ms": _percentiles(service),
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
              warmup_s: float, connections: int, seed: int = 1234,
              timeout_s: float = 30.0) -> Result:
    result = Result(target_rps=target_rps, duration_s=duration_s)
    limits = httpx.Limits(max_connections=connections,
                          max_keepalive_connections=connections)
    rng = random.Random(seed)
    in_flight = [0]

    async with httpx.AsyncClient(limits=limits, timeout=timeout_s) as client:
        # Warm-up traffic is sent but not measured: the first requests to a fresh
        # worker pay allocation costs that are not representative of steady state.
        warm = Result(target_rps=target_rps, duration_s=warmup_s)
        await _drive(client, url, payloads, target_rps, warmup_s, warm, rng, in_flight)
        await _drive(client, url, payloads, target_rps, duration_s, result, rng, in_flight)
    return result


def _run_one_process(args: tuple) -> tuple[list[tuple], int, int, int, float]:
    """One generator process's share of the offered rate.

    A module-level function because ProcessPoolExecutor has to pickle it. Samples come back
    as plain tuples for the same reason; the parent rebuilds them so the percentiles are
    computed over the whole run rather than averaged per process.
    """
    url, payloads, rps, duration_s, warmup_s, connections, seed, timeout_s = args
    result = asyncio.run(run(url, payloads, rps, duration_s, warmup_s, connections,
                             seed=seed, timeout_s=timeout_s))
    samples = [(s.due, s.sent, s.finished, s.status, s.server_ms) for s in result.samples]
    return samples, result.errors, result.max_in_flight, result.offered, result.span_s


def run_sweep_point(url: str, payloads: list[bytes], target_rps: float, duration_s: float,
                    warmup_s: float, connections: int, processes: int,
                    max_send_lag_ms: float, timeout_s: float) -> Result:
    """One target rate, offered by `processes` generators at once.

    One event loop cannot schedule much past a couple of hundred arrivals per second while
    sharing four cores with the server, and when it falls behind it charges its own lateness
    to the server (see the module docstring). Splitting the rate is what makes a capacity
    number about the service.

    Each process gets its own seed, or they would all send the same arrival pattern at the
    same instants and the aggregate would be N synchronised bursts rather than a Poisson
    process at N times the rate.
    """
    merged = Result(target_rps=target_rps, duration_s=duration_s, processes=processes,
                    max_send_lag_ms=max_send_lag_ms, connections=connections,
                    loadavg_1m=round(os.getloadavg()[0], 2))
    share = target_rps / processes
    # `connections` is per generator process, not a budget to divide. Dividing it made the
    # client's own pool the queue: three processes with 21 connections each peaked at 235
    # requests in flight, and the wait for a connection sits inside the measured service
    # time - so the server read 77 ms while its own handler header said 18 ms.
    jobs = [
        (url, payloads, share, duration_s, warmup_s, connections, 1234 + index, timeout_s)
        for index in range(processes)
    ]

    if processes == 1:
        outcomes = [_run_one_process(jobs[0])]
    else:
        # multiprocessing.Pool rather than ProcessPoolExecutor, because its workers are
        # daemonic: interrupt the sweep and they go with it. Executor children are not, and
        # a Ctrl-C left three orphaned generators firing at the server for nine minutes -
        # which then showed up as unexplained load in the next measurement.
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            outcomes = list(pool.map(_run_one_process, jobs))

    for samples, errors, in_flight, offered, span in outcomes:
        merged.samples.extend(
            Sample(due=due, sent=sent, finished=finished, status=status, server_ms=server)
            for due, sent, finished, status, server in samples
        )
        merged.errors += errors
        # Summed, not maxed: concurrency in flight is a property of the whole client side,
        # and each process only ever saw its own share.
        merged.max_in_flight += in_flight
        merged.offered += offered
        # The longest process's wall time, because they ran concurrently. Each process's
        # own perf_counter is used inside it and never compared against another's.
        merged.span_s = max(merged.span_s, span)
    return merged


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

    result.offered += len(tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    # Includes the drain: the clock runs until the last response, not until the last
    # request was due.
    result.span_s = max(result.span_s, time.perf_counter() - start)


def load_payloads(path: Path | None, count: int) -> list[bytes]:
    """Distinct users, so the run does not measure one perfectly cached code path."""
    from bl_ranking.serving.ranker import WARMUP_USER

    if path is not None:
        # A named file that is not there used to fall through to the synthetic payloads,
        # so a typo produced a healthy-looking report for traffic nobody asked about.
        if not path.exists():
            raise SystemExit(
                f"--payloads {path} does not exist. Omit the flag to use the built-in "
                f"synthetic payloads; naming a file that is not there would otherwise be "
                f"reported as a successful run of a different workload."
            )
        records = json.loads(path.read_text())
        if not records:
            raise SystemExit(f"--payloads {path} contains no records.")
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
    """The report. Service latency is the headline; send lag says whether to believe it."""
    lines = [
        "",
        f"{'target':>7} {'achv':>7} {'ok':>7} {'err':>4} "
        f"{'svc p50':>8} {'svc p99':>8} {'svc max':>9} "
        f"{'lag p50':>8} {'lag p99':>9} {'clnt p50':>9} {'clnt p99':>10}  verdict",
        "-" * 116,
    ]
    for s in summaries:
        service, lag, client = s["service_ms"], s["send_lag_ms"], s["client_ms"]
        if not service:
            lines.append(f"{s['target_rps']:>7.0f} {'-':>7} {s['ok']:>7} {s['errors']:>4}"
                         f"   no successful responses")
            continue
        verdict = ("GENERATOR " + "+".join(s["saturation_reasons"]).upper()
                   if s["generator_saturated"] else "ok")
        lines.append(
            f"{s['target_rps']:>7.0f} {s['achieved_rps']:>7.1f} {s['ok']:>7} "
            f"{s['errors']:>4} {service['p50']:>8.2f} {service['p99']:>8.2f} "
            f"{service['max']:>9.2f} {lag['p50']:>8.2f} {lag['p99']:>9.2f} "
            f"{client['p50']:>9.2f} {client['p99']:>10.2f}  {verdict}"
        )
    lines += [
        "",
        "all times in ms.  svc = sent -> response.  lag = due -> sent (the generator's own",
        "lateness).  clnt = due -> response, which is svc + lag + queueing.",
        "",
    ]
    for s in summaries:
        if s["server_ms"]:
            lines.append(
                f"  at {s['target_rps']:>5.0f} rps  x{s['processes']} generator(s), "
                f"offered {s['offered']}, in flight up to {s['max_in_flight']}, "
                f"handler header p50 {s['server_ms']['p50']:.2f} p99 "
                f"{s['server_ms']['p99']:.2f}"
            )
    if any(s["generator_saturated"] for s in summaries):
        lines += [
            "",
            "A row marked GENERATOR ... describes this program, not the service.",
            "  LATE   its p99 send lag is above --max-send-lag-ms: the scheduling loop fell",
            "         behind, so client latency is mostly our own lateness.",
            "  QUEUED far more requests are in flight than there are connections to carry",
            "         them, so they wait in the client's pool - and that wait is inside the",
            "         service time, which makes the server look slow. One generator at 300",
            "         rps reported 17.8 s; three reported 20 ms, same server, same moment.",
            "Raise --processes (and --connections) and re-run before quoting such a row.",
        ]
    return "\n".join(lines)


def _target_rates(text: str) -> list[float]:
    """Parse --rps, refusing the values that used to make the harness misbehave.

    A negative rate gave `expovariate` a negative mean, so every arrival was due *before*
    the last one and the scheduling loop allocated tasks as fast as it could: 4.5 GB
    resident in 51 seconds, on a flag typo.
    """
    rates = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rate = float(part)
        except ValueError:
            raise SystemExit(f"--rps: {part!r} is not a number") from None
        if not rate > 0 or rate != rate or rate == float("inf"):
            raise SystemExit(f"--rps: {part!r} must be a positive, finite rate")
        rates.append(rate)
    if not rates:
        raise SystemExit("--rps: no rates given")
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080/rank")
    parser.add_argument("--rps", default="50,100,200,400",
                        help="comma-separated target rates to sweep")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=4.0)
    parser.add_argument("--connections", type=int, default=128,
                        help="keep-alive connections PER generator process. Enough that "
                             "requests never wait for one: that wait is inside the measured "
                             "service time and reads as the server being slow")
    parser.add_argument("--processes", type=int, default=1,
                        help="generator processes sharing the offered rate. One event loop "
                             "cannot schedule much past ~200 rps on a small box, and when "
                             "it falls behind it charges its own lateness to the server")
    parser.add_argument("--max-send-lag-ms", type=float, default=100.0,
                        help="a run whose p99 send lag exceeds this is reported as "
                             "generator-saturated, because its latencies are this "
                             "program's backlog rather than the service's")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-request HTTP timeout in seconds")
    parser.add_argument("--payloads", type=Path, default=None)
    parser.add_argument("--payload-count", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.processes < 1:
        raise SystemExit("--processes must be at least 1")
    payloads = load_payloads(args.payloads, args.payload_count)

    summaries = []
    for rps in _target_rates(args.rps):
        result = run_sweep_point(args.url, payloads, rps, args.duration, args.warmup,
                                 args.connections, args.processes, args.max_send_lag_ms,
                                 args.timeout)
        summaries.append(result.summary())
        print(f"  {rps:>6.0f} rps done: {summaries[-1]['ok']} ok, "
              f"{summaries[-1]['errors']} errors"
              f"{', GENERATOR SATURATED' if summaries[-1]['generator_saturated'] else ''}",
              flush=True)

    report = render(summaries)
    print(report)
    if args.out:
        # JSON and the human report go to separate files: mixing them makes the JSON
        # unparseable by anything downstream, which is the whole reason to emit it.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2) + "\n")
        args.out.with_suffix(".txt").write_text(report + "\n")
        print(f"\nwritten to {args.out} and {args.out.with_suffix('.txt')}")

    # Non-zero when the run produced nothing to report, so `make loadtest` against a server
    # that is down or broken fails instead of quietly replacing the published results with a
    # table of zeros and exiting 0.
    dead = [s for s in summaries if s["ok"] == 0]
    if dead:
        raise SystemExit(
            f"no successful responses at {', '.join(str(s['target_rps']) for s in dead)} "
            f"rps - is {args.url} up and ready? (GET /readyz)"
        )


if __name__ == "__main__":
    main()
