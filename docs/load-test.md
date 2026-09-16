# Load simulation

Reproduce with:

```bash
make serve                      # in one terminal
make loadtest                   # in another: the sweep, three times over for the tables below

# and the two single-rate holds, which are what separate a steady state from a backlog
# that had not filled yet:
python loadtest/run_load.py --rps 250 --duration 60 --warmup 5 --processes 3
python loadtest/run_load.py --rps 300 --duration 60 --warmup 5 --processes 3
```

## Method

Arrivals are scheduled on a **Poisson process at a fixed target rate**, independent of
how the server is doing. That is the whole point.

A closed-loop client — a pool of workers that each send, wait for the reply, and send
again — cannot do this. When the server slows down, the client slows with it and simply
stops offering load, so the requests that *would* have arrived during a stall are never
sent and never appear in the percentiles. This is coordinated omission, and it makes a
struggling service look healthiest exactly when it is worst.

So:

- **Latency is measured from the moment a request was due**, not from when it was sent.
  Queueing is inside the number, which is what the user on the landing page experiences.
- **The server's own handler time** comes back in `X-Process-Time-Ms` and is reported
  alongside. The gap between the two *is* the queueing, and it is the first thing to look
  at when p99 leaves p50 behind.
- **Percentiles are nearest-rank** on the sorted sample, so every value printed is a real
  observation rather than an interpolation between two.
- **The sweep matters more than any single point.** The rate at which *achieved* rps
  falls behind *target* is the capacity estimate; percentiles measured past that point
  describe an overloaded system, not the service's latency.
- A warm-up period is sent and discarded: the first requests into a fresh worker pay
  allocation costs that are not steady state.

### Three latencies, and why the report prints all of them

An open-loop generator can be wrong in a way a closed-loop one cannot: it can fail to
keep up with its own schedule and then charge its lateness to the server. So each row
separates

| column | measures | meaning |
|---|---|---|
| `lag` | due → sent | the generator's own lateness. **The validity check** — if this is not small, nothing else on the row means anything |
| `svc` | sent → response | what the server and the network did |
| `clnt` | due → response | `svc` + `lag` + queueing: what a user would have experienced *if* the generator kept up |

and a row is marked `GENERATOR LATE` or `QUEUED` rather than reported as latency when the
send lag exceeds `--max-send-lag-ms` or far more requests are in flight than there are
connections to carry them. Both are worth stating because both were measured here:

- **One event loop cannot schedule much past ~200 arrivals per second on this box.** At a
  300 rps target, one generator reported svc p50 22.3 s and 99 rps achieved; three
  generators, against the same server in the same minute, reported 20 ms and 293 rps.
- **A pool of 128 connections per process was not enough at 400 rps.** Requests queued
  inside the client, the queue compounded — 2,364 in flight against 384 sockets — and the
  row read svc p50 1.6 s, p99 18.2 s. Raising the pool to 384 moved the same run to p50
  45 ms, so most of those 18 seconds were the client's. The default is now 384: an idle
  keep-alive socket costs nothing and a misattributed 18 seconds costs a capacity decision.
  It did not make 400 rps measurable here — the generator still cannot offer that rate on
  four shared cores — it just stopped the client's queue from being read as the server's.

A row whose responses are mostly not 2xx is marked too, for the same reason: percentiles
of refusals look excellent. That check exists because it caught a real fault in this
harness — the synthetic payloads drew `session_dt` and `register_date` on independent
days, so half of them had the survey submitted before the session that showed it, which
the schema correctly refuses. The sweep measured a service answering 422s at half the
offered rate and called it capacity.

**Stated caveat.** The generator runs on the same machine as the server, so above a few
hundred requests per second the two compete. Numbers past that point understate what the
service would do on its own hardware. They are reported anyway, marked, rather than
quietly trimmed.

## Configuration under test

| | |
|---|---|
| hardware | 12 vCPU, 16 GB (macOS 14.2, x86_64) |
| workers | 3 uvicorn processes, 1 BLAS thread each |
| server | uvloop, httptools, access log off, ORJSON responses |
| feature path | `fast` (the vectorised path; equivalence-tested against the research one) |
| payout backend | `surrogate` — the shipped default: a CatBoost student distilled from the TabPFN teacher (`catboost_fallback` is within noise of it) |
| model | version 1, 10 brands. 128,442 rows ingested, 62,286 with enough survey answers to preprocess, 55,368 fitted and 6,918 held out by the 7-day split |
| brands per request | 10 |
| payloads | 200 distinct users, so the run is not one perfectly cached code path |
| generators | 3 processes, 384 connections each |

## Results

Three identical sweeps — `make loadtest`, 30 s per rate after a 5 s warm-up — because one
sweep is not a measurement on a box this small. Each cell is the median of the three with
the full spread beside it. Client-side, measured from each request's scheduled arrival time
so queueing is inside the number; handler time is from the `X-Process-Time-Ms` header. All
values in milliseconds, and no request errored anywhere in any of the three runs.

| target rps | achieved | clnt p50 | clnt p99 | handler p50 | runs usable |
|---|---|---|---|---|---|
| 25 | 25.0 | 9.66 (9.48–10.17) | 22.27 (16.28–30.04) | 8.68 | 3 of 3 |
| 50 | 48.7 | 9.46 (8.86–10.60) | 19.62 (19.22–43.64) | 8.45 | 3 of 3 |
| 100 | 98.6 | 10.04 (9.96–12.24) | 55.87 (27.43–56.03) | 9.30 | 3 of 3 |
| 200 | 196.0 | 11.34 (9.38–12.78) | 55.38 (35.34–59.94) | 10.62 | 3 of 3 |
| 300 | 293.3 | 12.50 (12.15–13.01) | 80.72 (61.00–121.55) | 11.72 | 3 of 3 |
| 350 | 343.4 | 12.11 (10.97–15.60) | 98.39 (28.64–**510**) | 11.40 | 3 of 3 |
| 400 | 365–394 | 11.3 / 17.9 | 80.45 / **3534** | 10.6 / 16.8 | 2 of 3 |

The 400 rps row shows its two usable runs rather than a median: the third could not offer
the rate at all — its generator fell behind (p99 send lag 335 ms) and the harness marked
it `GENERATOR LATE` rather than charging the backlog to the server. Of the two that did
offer it, one held p99 at 80 ms and the other blew out to 3.5 s, which is the signature of
a rate at the edge, not a capacity number.

And two single rates held for a full minute rather than 30 s, to separate a transient
backlog from a steady state:

| rate | duration | achieved | clnt p50 | clnt p99 | handler p50 |
|---|---|---|---|---|---|
| 250 | 60 s | 247.3 | 13.70 | 164.38 | 13.05 |
| 300 | 60 s | 298.4 | 10.42 | 68.51 | 9.79 |

## Reading it

**Up to 300 rps the service is flat and boring, which is the good outcome.** p50 sits
between 9.5 and 12.5 ms at every rate from 25 to 300, p99 under 81 ms, handler time 8.4 to
11.7 ms, achieved rate equal to target, nothing erroring — and the 60-second hold at 300 rps
(p50 10.4 ms, p99 68.5 ms) looks the same as the 30-second sweep, so it is a steady state
and not a queue that had not filled yet.

**350 rps mostly holds, but the spread is starting to show.** All three sweeps delivered
343 rps at p50 11–16 ms, and two of them kept p99 under 100 ms — but one came back at p99
510 ms with nothing erroring, the queue simply building and draining. A rate whose p99 moves
by 5× between identical runs is the edge announcing itself.

**400 rps is that edge.** Two of three sweeps offered it (365 and 394 rps achieved) and
split hard: one held p99 at 80 ms, the other blew out to 3.5 s. The third sweep could not
offer 400 at all — its *generator* fell behind its own schedule (p99 send lag 335 ms), and
the harness marked that row `GENERATOR LATE` rather than charging the backlog to the server,
which is the whole reason the send lag is measured. A rate that swings 80 ms → 3.5 s across
runs, and that the generator itself struggles to produce, is a rate to plan below.

**So: plan on 300 rps per three-worker box, with 350 as headroom you do not lean on.** Both
are floors rather than ceilings — the generator shares the machine, so on dedicated hardware
the service would go further — but a floor measured three times is worth more than a ceiling
measured once. The way to get more is more worker processes on more cores, because the work
is CPU-bound and scales by process.

**Operating point.** At 100 rps — comfortably inside capacity — p50 is 10.0 ms and p99 is
55.9 ms end to end, with the handler itself at 9.3 ms p50. The two gaps are worth keeping
apart. `X-Process-Time-Ms` is stamped by the outermost middleware, so it already contains
body parsing and Pydantic validation: against ~2.7 ms of scoring measured in-process, the
other ~6.6 ms of that 9.3 ms is the HTTP layer — reading the body and validating 22 fields.
The ~0.7 ms between the header and the client figure is loopback, connection handling and OS
queueing, of which ~0.6 ms is the generator's own send lag.

**What earlier versions of this document got wrong, twice.** The first reported capacity as
"between 200 and 300 rps, and the failure is a cliff", with p50 going to 6.6 s at a 300 rps
target — that was a single generator process falling behind its own schedule and billing the
wait to the server. The second, on the same small box, swung the other way and read a single
20-second run as 387 rps at 47 ms. Both were one run. The answer that survives repetition is
the modest one, and the reason to publish the spread rather than the best cell is that the
best cell is what both mistakes had in common.

## Where the time goes

In-process, models warm, one request = one user scored against 10 brands, against the
production model above. Reproduce with `python scripts/profile_request.py`:

| stage | p50 | p95 | p99 |
|---|---|---|---|
| build the user's features (`fast_features.build_feature_row`) | 0.083 ms | 0.147 ms | 0.196 ms |
| broadcast across the 10 brands (`batch.from_row`) | 0.039 ms | 0.060 ms | 0.081 ms |
| CatBoost `Pool` construction, shared by both models | 0.382 ms | 0.624 ms | 0.760 ms |
| classifier `predict_proba` | 1.269 ms | 1.741 ms | 2.051 ms |
| payout `predict` | 0.830 ms | 1.118 ms | 1.280 ms |
| sort, rank, serialise | 0.078 ms | 0.163 ms | 0.192 ms |
| **total in-process** | **2.786 ms** | **3.382 ms** | **3.823 ms** |
| the same work through `ranker.rank()` | 2.564 ms | 3.349 ms | 3.501 ms |

The last row is the check that the parts add up. The sum of the separately-timed stages
(2.786 ms) runs about 0.22 ms *above* the single `ranker.rank()` call (2.564 ms) — the cost
of the per-stage timing itself, not a discrepancy. Three runs of the script gave a total p50
of 2.786, 2.705 and 2.676 ms, a spread of 110 µs, which is what to assume on any figure here.

Two thirds of the request is the two model calls, which is the right shape — there is no
pandas overhead left to remove. Repeating a *single* user instead of 200 distinct ones
gives 2.0 ms, the friendliest possible case for cache locality and the figure to use when
comparing against a micro-benchmark rather than against traffic.

The same request through the unmodified research pipeline — with the models already warm,
so this excludes the per-request model loading it would also do — is **66.8 ms p50**, 26×
the served path. The difference is pandas per-operation overhead, not arithmetic;
`serving/fast_features.py` explains it line by line. (An earlier version of this document
quoted 4.1 ms for the served path: that was measured before the gender lookup stopped
materialising its table through `to_pylist()` on every load, and is not what the code does
now — the 200-user figure with the fix in place is 2.7 ms.)

## Payout backends

The choice of payout backend dominates everything else, which is why the surrogate exists:

| backend | payout prediction | total request |
|---|---|---|
| `surrogate` / `catboost_fallback` | 0.83 ms | 2.7 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=4` (exact, shipped teacher) | ~259 ms | 261 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=2` | 150 ms | ~152 ms |
| `tabpfn_local`, `fit_preprocessors`, `n_estimators=2` | 7,650 ms | ~7,652 ms |
| `tabpfn_client` | one network round trip | request + RTT |

The surrogate/teacher rows over 200 distinct users come from `scripts/compare_backends.py`
(surrogate p50 2.21 ms, teacher p50 261.3 ms, a 118× gap). The `n_estimators=2` rows come
from the `fit_mode` comparison, which holds the ensemble size fixed so that the only variable
is the fit mode. See the README section "Productizing TabPFN" for how those were measured and
what the trade-off costs in accuracy.
