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
| payout backend | `catboost_fallback` (no network, so the numbers isolate our own code) |
| brands per request | 15 |
| payloads | 200 distinct users, so the run is not one perfectly cached code path |

## Results

_Populated by `make loadtest` against a running endpoint; see the
`loadtest/results/` files it writes._

## Reading it

_The capacity reading goes here: the target rate at which achieved rps
first falls behind, which is the number that bounds the service._

## Where the time goes

In-process, models warm, one request = one user scored against 15 brands:

| stage | p50 |
|---|---|
| feature construction (`fast_features.build_feature_row`) | 0.014 ms |
| broadcast to 15 brand rows | 0.015 ms |
| CatBoost `Pool` construction (shared by both models) | 0.35 ms |
| classifier `predict_proba` | 0.70 ms |
| payout `predict` | 0.48 ms |
| sort, rank, serialise | ~0.2 ms |
| **total in-process** | **~1.8 ms** |

For comparison, the same request through the unmodified research pipeline — with the
models already warm, so this excludes the per-request model loading it would also do —
is **54 ms p50**. The difference is pandas per-operation overhead, not arithmetic;
`serving/fast_features.py` explains it line by line.

The remaining gap between 1.8 ms in-process and the HTTP p50 is FastAPI request
validation, response serialisation and the ASGI stack. Measured on the same box, a
handler that does nothing (`/healthz`) costs 0.41 ms server-side, which is the floor.

## Payout backends

The table above uses `catboost_fallback` so the numbers reflect our own code rather than
a third party's. The choice of payout backend dominates everything else:

| backend | payout prediction | total request |
|---|---|---|
| `surrogate` / `catboost_fallback` | ~0.5 ms | ~1.8 ms |
| `tabpfn_local`, `fit_with_cache` | 260 ms | ~261 ms |
| `tabpfn_local`, `fit_preprocessors` | 8,690 ms | ~8,691 ms |
| `tabpfn_client` | one network round trip | request + RTT |

See the README section "Productizing TabPFN" for how those were measured and what the
trade-off costs in accuracy.
