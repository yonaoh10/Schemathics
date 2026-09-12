# Load simulation

Reproduce with:

```bash
make serve                      # in one terminal
make loadtest                   # in another
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

**Stated caveat.** The generator runs on the same four cores as the server, so above a few
hundred requests per second the two compete. Numbers past that point understate what the
service would do on its own hardware. They are reported anyway, marked, rather than
quietly trimmed.

## Configuration under test

| | |
|---|---|
| hardware | 4 vCPU, 15 GB |
| workers | 3 uvicorn processes, 1 BLAS thread each |
| server | uvloop, httptools, access log off, ORJSON responses |
| feature path | `fast` (the vectorised path; equivalence-tested against the research one) |
| payout backend | `surrogate` — the shipped default: a CatBoost student distilled from the TabPFN teacher (`catboost_fallback` is within noise of it) |
| model | version 3, 15 brands. 81,002 rows ingested, 73,492 with enough survey answers to preprocess, 65,727 fitted and 7,765 held out by the 7-day split |
| brands per request | 15 |
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
| 25 | 25.0 | 10.6 (10.1–10.9) | 17.9 (16.6–21.4) | 6.8 | 3 of 3 |
| 50 | 48.7 | 9.7 (8.9–10.8) | 17.8 (16.4–21.1) | 6.4 | 3 of 3 |
| 100 | 98.6 | 9.8 (9.0–10.0) | 21.3 (20.9–22.0) | 6.6 | 3 of 3 |
| 200 | 196.0 | 10.4 (10.2–13.7) | 33.2 (27.9–47.3) | 6.8 | 3 of 3 |
| 300 | 293.2 (288–293) | 15.0 (13.8–40.7) | 53.6 (48.0–**750**) | 10.5 | 3 of 3 |
| 350 | 343.2 (2 of 3) | 38.6 (31.5–38.6) | 326 (129–326) | 29.2 | 2 of 3 |
| 400 | — | — | — | 41.4 | **0 of 3** |

And two single rates held for a full minute rather than 30 s, to separate a transient
backlog from a steady state:

| rate | duration | achieved | clnt p50 | clnt p99 | handler p50 |
|---|---|---|---|---|---|
| 250 | 60 s | 247.3 | 11.5 | 48.8 | 7.5 |
| 300 | 60 s | 298.4 | 17.8 | 64.9 | 12.8 |

## Reading it

**Up to 250 rps the service is flat and boring, which is the good outcome.** p50 sits
between 9.7 and 11.5 ms at every rate from 25 to 250, p99 under 50 ms, handler time 6.4 to
7.5 ms, achieved rate equal to target, nothing erroring — and the 60-second hold at 250 rps
looks the same as the 30-second one, so it is a steady state and not a queue that had not
filled yet.

**300 rps is the edge, and the spread says so.** Two sweeps put it at p50 14–15 ms and p99
48–54 ms, and a 60-second hold at 300 agreed (17.8 / 64.9). The third sweep, same command
on the same box, came back at p50 41 ms and p99 750 ms with the handler itself at 24 ms.
Nothing errored in that run either; the queue simply built and drained. A rate whose p99
moves by 14× between identical runs is a rate to plan below, not a capacity figure.

**350 rps is where this box stops being able to ask the question.** The service still
delivered 343 rps in two runs of three, at p50 31–39 ms and p99 129–326 ms. In the third the
*generator* fell behind its own schedule — 3 server workers and 3 generator processes want
more than four cores — and the row was marked `GENERATOR LATE` rather than reported. At 400
rps that happened in all three runs, so the table has no latency for it at all. What the
handler header still shows is the server absorbing the contention: 41 ms p50 against 6.8 ms
at 200 rps.

**So: plan on 250 rps per three-worker box, with 300 as the headroom you do not use.** Both
numbers are floors rather than ceilings — the generator is on the same four cores, and on
dedicated hardware the service would go further — but a floor measured three times is worth
more than a ceiling measured once. The way to get more is more worker processes on more
cores, because the work is CPU-bound and scales by process.

**Operating point.** At 100 rps — comfortably inside capacity — p50 is 9.8 ms and p99 is
21.3 ms end to end, with the handler itself at 6.6 ms p50. The two gaps are worth keeping
apart. `X-Process-Time-Ms` is stamped by the outermost middleware, so it already contains
body parsing and Pydantic validation: against 2.8 ms of scoring measured in-process, about
half of that 6.6 ms is the HTTP layer. The 3.2 ms beyond the header is loopback, connection
handling and OS queueing across three workers sharing four cores with the load generator, of
which 0.8 ms is the generator's own send lag.

**What earlier versions of this document got wrong, twice.** The first reported capacity as
"between 200 and 300 rps, and the failure is a cliff", with p50 going to 6.6 s at a 300 rps
target — that was a single generator process falling behind its own schedule and billing the
wait to the server. The second, after the harness grew three generator processes and learned
to report its own send lag, swung the other way and read a single 20-second run as 387 rps
at 47 ms. Both were one run. The answer that survived repetition is the modest one in this
section, and the reason to publish the spread rather than the best cell is that the best cell
is what both mistakes had in common.

## Where the time goes

In-process, models warm, one request = one user scored against 15 brands, against the
production model above. Reproduce with `python scripts/profile_request.py`:

| stage | p50 | p95 | p99 |
|---|---|---|---|
| build the user's features (`fast_features.build_feature_row`) | 0.063 ms | 0.103 ms | 0.113 ms |
| broadcast across the 15 brands (`batch.from_row`) | 0.030 ms | 0.042 ms | 0.070 ms |
| CatBoost `Pool` construction, shared by both models | 0.365 ms | 0.441 ms | 0.485 ms |
| classifier `predict_proba` | 1.319 ms | 1.500 ms | 1.676 ms |
| payout `predict` | 0.910 ms | 1.015 ms | 1.048 ms |
| sort, rank, serialise | 0.069 ms | 0.108 ms | 0.124 ms |
| **total in-process** | **2.766 ms** | **3.111 ms** | **3.313 ms** |
| the same work through `ranker.rank()` | 2.809 ms | 3.039 ms | 3.478 ms |

The last row is the check that the parts add up: timing the stages individually and timing
the whole call agree to within 50 µs. Two runs of the script minutes apart gave 2.766 and
2.777 ms, which is the spread to assume on any figure here.

Two thirds of the request is the two model calls, which is the right shape — there is no
pandas overhead left to remove. Repeating a *single* user instead of 200 distinct ones
gives 2.411 ms, the friendliest possible case for cache locality and the figure to use when
comparing against a micro-benchmark rather than against traffic.

The same request through the unmodified research pipeline — with the models already warm,
so this excludes the per-request model loading it would also do — is **53.7 ms p50**, 19×
the served path. The difference is pandas per-operation overhead, not arithmetic;
`serving/fast_features.py` explains it line by line.

Over 200 distinct users this path measured 4.1 ms before the gender lookup stopped
materialising its table through `to_pylist()` on every load and started reading the Arrow
dictionary directly. The repeated-user figure was unchanged by that fix at 2.41 ms, which
is exactly the signature of a cost paid per *distinct* name.

## Payout backends

The choice of payout backend dominates everything else, which is why the surrogate exists:

| backend | payout prediction | total request |
|---|---|---|
| `surrogate` / `catboost_fallback` | 0.91 ms | 2.77 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=4` (exact) | ~455 ms | 457 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=2` | 260 ms | ~262 ms |
| `tabpfn_local`, `fit_preprocessors`, `n_estimators=2` | 8,690 ms | ~8,692 ms |
| `tabpfn_client` | one network round trip | request + RTT |

The 457 ms is end to end over 200 users through the whole ranker; the `n_estimators=2` rows
come from the `fit_mode` comparison, which holds the ensemble size fixed so that the only
variable is the fit mode. See the README section "Productizing TabPFN" for how those were
measured and what the trade-off costs in accuracy.
