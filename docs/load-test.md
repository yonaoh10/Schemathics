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

**Stated caveat.** The generator runs on the same four cores as the server, so above a
few hundred requests per second it becomes a co-bottleneck. Numbers past that point
understate what the service would do on its own hardware. They are reported anyway,
marked, rather than quietly trimmed.

## Configuration under test

| | |
|---|---|
| hardware | 4 vCPU, 15 GB |
| workers | 3 uvicorn processes, 1 BLAS thread each |
| server | uvloop, httptools, access log off, ORJSON responses |
| feature path | `fast` (the vectorised path; equivalence-tested against the research one) |
| payout backend | `surrogate` (the configured default; no network at request time) |
| brands per request | 15 |
| payloads | 200 distinct users, so the run is not one perfectly cached code path |

## Results

Client-side, measured from each request's scheduled arrival time so queueing is inside
the number. All values in milliseconds.

| target rps | achieved | ok | errors | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 24.2 | 483 | 0 | 10.4 | 13.2 | 15.9 | 25.3 | 40.3 | 40.3 |
| 50 | 48.5 | 968 | 0 | 10.5 | 14.5 | 16.2 | 19.8 | 46.3 | 46.3 |
| 100 | 97.6 | 1,951 | 0 | 10.6 | 14.7 | 16.8 | 38.1 | 109.3 | 122.3 |
| 200 | 197.7 | 3,956 | 0 | 21.7 | 37.8 | 43.8 | 60.8 | 85.3 | 118.8 |
| 300 | **130.4** | 5,942 | 0 | 6,572 | 28,206 | 31,190 | 35,343 | 38,516 | 40,375 |
| 400 | **73.1** | 8,019 | 0 | 58,020 | 93,247 | 97,645 | 103,880 | 107,343 | 109,267 |

Server-side handler time for the same runs, from the `X-Process-Time-Ms` header:

| target rps | handler p50 | handler p99 | client p99 minus handler p99 |
|---|---|---|---|
| 25 | 6.8 | 13.9 | +11.4 |
| 50 | 7.0 | 14.1 | +5.7 |
| 100 | 7.0 | 20.8 | +17.3 |
| 200 | 16.5 | 48.5 | +12.3 |
| 300 | 22.8 | 68.9 | +35,274 |
| 400 | 38.3 | 155.1 | +103,725 |

That last column is the point of measuring both. Up to 200 rps the gap is tens of
milliseconds of ordinary queueing. At 300 the handler still looks healthy at 23 ms while
users are waiting 35 seconds — the work is fine, the arrivals are not being served.

## Reading it

**Capacity is between 200 and 300 requests per second, and the failure is a cliff.**
Achieved rate tracks target exactly to 200 rps (197.7 achieved), with p50 21.7 ms and
p99 60.8 ms. At a 300 rps target the server delivers only 130 rps and latency goes to
seconds; at 400 it delivers 73. Nothing errors — requests are accepted and then queue,
which is the worst failure shape for a page load and exactly what an open-loop test is
for. A closed-loop client would have reported 300 rps as "slower" rather than "broken",
because it would simply have stopped offering load.

**Operating point.** At 100 rps — comfortably inside capacity — p50 is 10.6 ms and p99
is 38.1 ms end to end, on three worker processes sharing four cores with the load
generator. The handler itself is 7.0 ms p50 there; the rest is queueing and the client.

**What this does and does not bound.** The generator runs on the same four cores as the
server, so the collapse point is a floor on capacity, not a ceiling: real capacity on
dedicated hardware is higher. It is reported as measured rather than adjusted upward.
The right response to needing more than 200 rps is more worker processes and more cores,
because the work is CPU-bound and scales by process.

## Where the time goes

In-process, models warm, one request = one user scored against 15 brands, against the
production model (81k training rows, 70 MB classifier):

| stage | p50 |
|---|---|
| build the user's 24 features (`fast_features.build_feature_row`) | 0.013 ms |
| broadcast across 15 brands | 0.014 ms |
| CatBoost `Pool` construction, shared by both models | 0.33 ms |
| classifier `predict_proba` | 0.97 ms |
| payout `predict` | 0.79 ms |
| sort, rank, serialise | 0.02 ms |
| **total in-process** | **2.41 ms** |

The same request through the unmodified research pipeline — with the models already
warm, so this excludes the per-request model loading it would also do — is **60.3 ms
p50**. The difference is pandas per-operation overhead, not arithmetic;
`serving/fast_features.py` explains it line by line.

Those stage figures repeat one user, which is the cleanest way to see where the time
sits but the friendliest possible case for cache locality. Over 200 *distinct* users the
same in-process call is 4.1 ms. Between that and ~7 ms server-side sits FastAPI request
validation, response serialisation, the ASGI stack, and three worker processes competing
for four cores with the load generator.

## Payout backends

The sweep above uses `catboost_fallback` so the numbers reflect our own code rather than
a third party's. The choice of payout backend dominates everything else:

| backend | payout prediction | total request |
|---|---|---|
| `surrogate` / `catboost_fallback` | 0.79 ms | 2.41 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=4` (shipped) | ~455 ms | 457 ms |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=2` | 260 ms | ~262 ms |
| `tabpfn_local`, `fit_preprocessors`, `n_estimators=2` | 8,690 ms | ~8,692 ms |
| `tabpfn_client` | one network round trip | request + RTT |

The 457 ms is end to end over 200 users through the whole ranker; the `n_estimators=2`
rows come from the `fit_mode` comparison, which holds the ensemble size fixed so that the
only variable is the fit mode. See the README section "Productizing TabPFN" for how those
were measured and what the trade-off costs in accuracy.
