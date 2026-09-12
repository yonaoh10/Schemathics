# Business Loans brand ranking — production system

Productization of two research scripts into a training pipeline, a model registry with
rollback, and a synchronous ranking endpoint.

When a user finishes the Business Loans funnel we have a few tens of milliseconds to
decide which lender brand goes first. Two models answer two different questions —
a CatBoost classifier for *will this brand accept the lead?* and a TabPFN regressor for
*how much would they pay?* — and their product orders the list:

```
expected_payout = P(brand accepts this user) x payout_if_accepted
```

The research code that fits and applies those models is treated as fixed. It is
vendored byte-for-byte in `src/bl_ranking/research/`, every training run records a
checksum of it, and a test asserts the digest has not moved. Everything else in this
repository is the machinery around it.

---

## Quick start

```bash
make setup          # venv + install
make all            # generate data -> Delta -> production + register -> train/test eval
make serve          # API on :8080
make loadtest       # open-loop latency sweep
```

`make all` finishes in a few minutes and needs no credentials and no network, because
the Makefile overrides the payout backend to `catboost_fallback`. The *configured*
default in `conf/config.yaml` is `surrogate`, which is the production recommendation;
the Makefile differs on purpose so a reviewer can get a working system before deciding
anything about TabPFN.

```bash
make train-prod BACKEND=surrogate        # the recommended default; needs `.[tabpfn]`
make train-prod BACKEND=tabpfn_local     # exact, slow on CPU
make train-prod BACKEND=tabpfn_client    # exact, hosted; needs TABPFN_TOKEN
```

`pip install -e '.[tabpfn]'` pulls torch. The weights are fetched on first use; to run
fully offline, pre-place `tabpfn-v2-regressor.ckpt` in `TABPFN_MODEL_CACHE_DIR`.

The whole stack in Docker, which is the closest local analogue of the Databricks
deployment:

```bash
docker compose -f docker/docker-compose.yml up -d --build mlflow api scheduler
# MLflow UI  http://localhost:5000
# API        http://localhost:8080/docs
```

### One request

```bash
curl -s localhost:8080/rank -H 'content-type: application/json' -d '{
  "session_dt":"2026-01-06 19:24:22","conversion_dt":"2026-01-06 19:26:10",
  "register_date":"2026-01-06 19:26:07","campaign_id":120227360861540306,
  "page":"top10us.com/app/business-loans-v2","auto_city":"Fort Lauderdale",
  "auto_country":"United States","auto_state":"Florida","device_type":"mobile",
  "sub1":1121993,"sub2":"01121993 Ad set","sub3":1513124082,
  "business_type":"C Corporation","credit_score":"Very Poor - Under 550",
  "industry":"construction","loan_amount":"$25,000 - $49,999",
  "loan_reason":"Equipment purchase","monthly_revenue":"$20,000 - $49,999",
  "time_in_business":"2+ years","fname":"Rigoberto","lname":"Rodriguez",
  "cellphone":"(786) 991-4030"}'
```

```json
{
  "ranking": {
    "sba central":     {"rank": 1.0, "expected_payout": 57.53},
    "forward funding": {"rank": 2.0, "expected_payout": 57.39},
    "fora financial":  {"rank": 3.0, "expected_payout": 52.68}
  },
  "meta": {
    "model_version": "fe0373d4f372...", "payout_backend": "surrogate",
    "payout_exact": false, "n_brands": 15, "latency_ms": 1.9
  }
}
```

`ranking` is exactly what the research function returns. `meta` says which model
version produced it and whether that backend is bit-identical to TabPFN.
`POST /rank/bare` returns the ranking alone, for drop-in compatibility.

---

## Architecture

```
  bl_full_data.csv                     conf/config.yaml          (one source of truth,
        |                                     |                   logged to MLflow)
        v                                     v
  data/ingest.py  --- quality gate ---> Delta table (versioned, time-travel)
        |                                     |
        |                       training/job.py  (MLflow run)
        |                                     |
        |            +------------------------+------------------------+
        |            |                                                 |
        |    train_test=True                                   train_test=False
        |    time split, per-day metrics,                      fit on all data,
        |    researcher log as artifact                        build the bundle
        |    nothing registered                                       |
        |                                                             v
        |                                          model version = ONE atomic bundle
        |                                          CB_bl_lead.cbm + payout context +
        |                                          all_clients.csv + gender table +
        |                                          manifest.json
        |                                                             |
        |                                               alias `champion` -> version N
        v                                                             |
  serving/app.py  <---- resolves models:/bl_brand_ranker@champion ----+
        |
        +-- /rank  /rank/bare  /healthz  /readyz  /model  /metrics
```

### Local stack vs Databricks

Everything runs locally, and each local piece has a one-to-one Databricks counterpart.
`databricks/databricks.yml` is the Asset Bundle that deploys it.

| Local | Databricks |
|---|---|
| `deltalake` (delta-rs) table on disk | Unity Catalog Delta table |
| MLflow server in Docker (sqlite + artifacts) | the workspace tracking server and UC registry |
| `bl_ranking.training.job` | the three tasks of the `bl_weekly_training` job |
| APScheduler in the `scheduler` container | the job's `schedule` block |
| FastAPI container | a Model Serving endpoint fronting the same pyfunc |
| `models:/bl_brand_ranker@champion` | `catalog.schema.bl_brand_ranker@champion` |

The schedule string lives in `conf/config.yaml` and is read by both, so they cannot
drift apart. A test asserts that.

---

## Training pipeline

Both research modes are preserved.

| | `train_test=True` | `train_test=False` |
|---|---|---|
| data | all but the last 7 days | everything |
| output | per-day classification reports, payout MAE/MAPE at four thresholds | the four research artifacts |
| registry | nothing | a new version, `champion` moved onto it |
| command | `make train` | `make train-prod` |

Every run logs the same technical parameters, so any two runs are comparable in the
MLflow UI regardless of mode:

- **data** — Delta table version, row and session counts, the window start and end,
  the lookback setting, and the ingest repair counters;
- **model** — every CatBoost hyper-parameter, the payout backend, the TabPFN context
  size, fit mode and pinned checkpoint;
- **provenance** — git SHA, a SHA-256 of the two vendored research scripts, Python and
  library versions, hostname.

Metrics are the same numbers the researcher log prints, captured as floats:
`clf.{accuracy,precision,recall,f1}` overall and per test day, and
`payout.{mae,mape}` overall and at the 5/10/15/20 thresholds.

**The researcher log is attached verbatim** as the `researcher_log` artifact. The
production wrapper calls `super()` before capturing anything, so that file is byte-for-
byte what the research code would have written on its own.

### What a full-scale run produces

81,002 rows over the two-month window, 74,353 of them registered sessions, 15 brands.
Evaluation mode, holding out the last 7 days:

| | |
|---|---|
| acceptance classifier | accuracy 0.744, precision 0.609, recall 0.579, F1 0.594 |
| positive rate in the test window | 32.3% |
| payout regressor | MAE $21.60, MAPE 27.3% over 3,200 paid rows |
| production run | 12 min; evaluation run 11 min; peak 9.5 GB resident |

These come from the synthetic extract, so the absolute values describe the generator as
much as the models. What they establish is that the pipeline runs end to end at real
volume, that both label classes are present in every test-day bucket, and that the
numbers land where the research code's own log puts them.

### Versioning and rollback

The research code writes four loose files. Loose files are a rollback hazard: copy the
CatBoost model and the payout context separately and a serving process can end up with
a classifier from one run and a payout context from another, with nothing to detect it.
`prediction_expected_payout` reorders the frame by the *payout* model's column list and
then feeds it to the *classifier*, so a mismatched pair mis-slots features silently.

So one training run produces one bundle, and one bundle is one MLflow model version.

```bash
make versions          # every registered version, marking the one being served
make rollback VERSION=3
docker compose -f docker/docker-compose.yml restart api
```

Moving the alias moves the classifier, the payout context, the brand universe and the
gender table together. `GET /model` reports what a live worker actually loaded, so
"which version is serving?" is answerable without guessing.

### Schedule

Weekly, Sunday 05:00. One definition, two consumers:

```yaml
schedule:
  cron: "0 0 5 ? * SUN *"     # Quartz: sec min hour day-of-month month day-of-week year
  timezone: UTC
```

Databricks takes the Quartz string directly. `bl_ranking.ops.schedule` parses the same
string into APScheduler fields for the local runner and refuses a 5-field unix cron,
which would otherwise mean something quite different. `make schedule` prints the
parsed result.

---

## Productizing TabPFN

This is the interesting part of the assignment, and the honest answer is uncomfortable.

**What the research code does.** TabPFN is in-context learning: `fit()` is not training,
it hands the model its 1000-row context. The research predictor calls `load_models()` —
and therefore `TabPFNRegressor().fit(context)` — inside `predict_()`, which runs on
**every request**. With `tabpfn-client` that means serialising the whole context to
parquet, uploading it, and posting a fit, before a single brand is scored. Then a second
round trip to predict. Per user. While the page is loading.

**The context is frozen for a week.** It only changes when Sunday's retrain changes it.
So the fit belongs in the weekly job, and a request should pay for its own ~15 rows and
nothing else. Everything below follows from that one observation.

### The `fit_mode` ladder

Measured on this box (4 CPU, 15 GB, context 1000x25, one request = 15 brand rows). All
three rows use `n_estimators=2`, so the only thing varying is `fit_mode`:

| `fit_mode` | start-up | payout prediction | max diff vs reference |
|---|---|---|---|
| `low_memory` | 0.2 s | 37.3 s | $4.03 |
| `fit_preprocessors` (library default) | 0.4 s | 8.69 s | reference |
| `fit_with_cache` | 14.9 s | **0.26 s** | **$0.000045** |

`fit_with_cache` is a **33x** improvement over the library default for predictions that
differ by 4.5e-5 on payouts of $60-140 — a relative error around 1e-6, far below any
plausible ranking tie. It is the single biggest exact-preserving win available, and it is
the one the research code leaves on the table. (`low_memory`, by contrast, is both
slowest *and* materially different, so it is never the right choice here.)

It does not rescue the request path. The shipped teacher configuration is
`fit_with_cache` with `n_estimators=4`, and a **complete request** through it — features,
classifier, payout, ranking — is **457 ms p50**, measured over 200 users. Against 2.41 ms
for the served path. So `tabpfn_local` is the right backend for batch scoring and for
teaching the surrogate, not for the funnel.

Per request, end to end, at the configuration each backend actually ships with:

| backend | per request | accuracy |
|---|---|---|
| `surrogate` (CatBoost student) | **4.1 ms** | approximate, measured and logged |
| `catboost_fallback` | ~4 ms | approximate; an availability path, not an accuracy claim |
| `tabpfn_client`, fit reused from the weekly job | one network round trip | exact |
| `tabpfn_local`, `fit_with_cache`, `n_estimators=4` | 457 ms | exact |

Both columns of that last table come from the same 200-user comparison, so they are
directly comparable. (The 2.41 ms figure quoted elsewhere repeats a single user, which
is the friendliest case for cache locality; 4.1 ms is the same code over 200 distinct
ones.)

### What the approximation costs

Two measurements, and the gap between them is the interesting part.

**At training time**, the student is compared against the teacher on held-out users
drawn from the distillation sample: rank correlation 0.993, dollar error 6.2%, and the
same top brand every time. That number is optimistic, and knowing why matters. The
distillation sample is built from the payout context — rows that were actually paid for
— so it is a narrower, higher-value slice than live traffic, and it compares *payout
predictions* rather than the ranking those predictions produce.

**End to end** is the number to quote. `scripts/compare_backends.py` scores 200
randomly generated users through the whole endpoint twice, once per backend, and
compares the rankings themselves:

| | surrogate vs TabPFN teacher |
|---|---|
| same brand in position 1 | 84.5% |
| same top 3, as a set | 82.0% |
| identical ordering of all 15 | 3.5% |
| **expected payout given up, averaged over all users** | **1.21%** |
| ... when the two disagree | 7.83% |
| ... p95 | 11.1% |
| ... worst single user | 27.8% |
| latency | 4.1 ms vs 456.7 ms (111x) |

Read the regret row, not the agreement rows. Agreement counts treat "picked a brand
worth two cents less" the same as "picked a much worse brand"; regret asks what the
student's choice is actually worth *under the teacher's own scores*. The student
disagrees about the first position for roughly one user in six, and when it does, it
gives up about 8% of that user's expected payout — 1.21% averaged across everyone.

**So the trade is 1.21% of expected payout per session against 450 ms of added page
latency.** Which side wins depends on the funnel's own latency-to-conversion curve,
which is a number the business has and I do not. The default is the surrogate because
450 ms on a landing page is very likely to cost more than 1.2%, but that is a judgement
the numbers above are meant to let someone else overturn — and
`BL_MODEL__PAYOUT__BACKEND=tabpfn_local` overturns it.

### The four backends

- **`surrogate`** (default). TabPFN labels a large sample offline at training time —
  the real training rows crossed with the brand universe, which is exactly the shape of
  an inference request — and a CatBoost student is fitted to those labels. Inference is
  microseconds. It is the only option that meets the stated latency requirement on CPU.
  Because it changes predictions, the distillation step measures itself against the
  teacher and logs the result, and `scripts/compare_backends.py` re-checks it end to end
  on the ranking rather than on the payout predictions behind it. Every response is
  tagged `payout_exact: false`.
- **`tabpfn_client`** — the research code's own hosted model, with the fit lifted out of
  the request path. The library supports this directly: `fit()` returns a server-side
  `fitted_train_set_id`, and an estimator whose `model_id_` is assigned can predict
  without ever fitting. The weekly job fits once, the id ships in the bundle, and a
  request costs one round trip carrying only its own rows. Same fitted set, same
  predictions, no upload. Two further fixes: the checkpoint is pinned, because the
  default lets the provider change payouts between two weekly runs with nothing in
  MLflow to explain it; and `TABPFN_CLIENT_CI_MODE=true` removes a spinner that
  busy-polls every 200 ms and quantises every call onto that grid.
- **`tabpfn_local`** — weights in-process, `fit_with_cache`, and the fitted estimator
  (including its KV cache) serialised into the bundle so replicas start warm with zero
  network. Exact and private. Use it on GPU, or for batch.
- **`catboost_fallback`** — no TabPFN at all. If the weights or the API are unreachable
  *when a worker loads its model*, that worker degrades to this and keeps answering. A
  slightly worse ranking beats a 503 while a user waits. Training does *not* degrade: a
  silently downgraded model would be registered and served for a week.

  The fallback is scoped to model load, not to each request: a worker that started
  healthy and then loses the hosted API mid-life will return 500s until it restarts.
  Closing that gap properly means a circuit breaker, not a silent per-request swap, and
  it is listed under "what I would do next" rather than pretended away.

Switch with one variable: `BL_MODEL__PAYOUT__BACKEND=tabpfn_client`.

The precedence is worth stating, because two reasonable rules compete. A bundle
records the backend it was built with, so rolling back to an older version brings that
version's backend with it rather than whatever the current config says. But an operator
who sets the variable is making a decision, and that wins over the bundle. So: explicit
override first, then the bundle's own backend, then the config default. `GET /model`
reports which one is actually in use.

Two smaller things worth knowing: TabPFN sends usage telemetry to a third party by
default (`TABPFN_DISABLE_TELEMETRY=1` turns it off, and the serving image sets it), and
the hosted client uploads the in-context rows — real leads, with name-derived and
geographic features — on every fit. For a lead-generation business that is a governance
question independent of latency.

---

## Serving

`POST /rank` takes one post-funnel dictionary and returns the ranked brands.

| endpoint | purpose |
|---|---|
| `POST /rank` | ranking plus the model identity that produced it |
| `POST /rank/bare` | the research dictionary alone |
| `GET /healthz` | liveness — the process is up |
| `GET /readyz` | readiness — models loaded **and** a real scoring call succeeded |
| `GET /model` | the serving version, backend, brand count |
| `GET /metrics` | Prometheus |

### What made it fast

Measured in-process against the production model (81k training rows, a 70 MB
classifier), models already warm, one request being one user scored against 15 brands:

| path | p50 | p95 | p99 |
|---|---|---|---|
| the research pipeline as written | 60.3 ms | 74.4 ms | 83.6 ms |
| what this system serves | **2.41 ms** | 2.68 ms | 3.61 ms |

25x, and every step of it exact-preserving. Where the 2.41 ms goes:

| stage | p50 |
|---|---|
| build the user's 24 features | 0.013 ms |
| broadcast them across 15 brands | 0.014 ms |
| build one CatBoost `Pool` | 0.33 ms |
| classifier `predict_proba` | 0.97 ms |
| payout `predict` | 0.79 ms |
| sort, rank, serialise | 0.02 ms |

Three changes got it there, in order of what they were worth:

1. **The feature pipeline runs over plain values instead of pandas.** The research
   pipeline expresses a 15-row transform as roughly a hundred pandas calls: a cross join
   (3.0 ms), a DataFrame construction from a dict (0.9 ms), four `to_datetime` calls
   (0.6 ms each), `Series.apply(lambda: pd.Series(...))` for the gender feature (1.3 ms),
   and about twenty `.loc[mask, col] = value` assignments across the four band mappings
   (0.33 ms each). That cost is per *operation*, not per row, so it does not shrink with
   the data. Feature construction is now 0.013 ms.
2. **Everything request-independent moved to start-up** — the CatBoost load, the payout
   context fit, the brand universe, the warning handler. The research predictor does all
   of it inside `predict_()`, on every request.
3. **One `Pool` per request, shared by both models.** They were fitted on the same
   columns with the same categorical set, so building the input twice was pure waste.

`serving/fast_features.py` is the same transformation over plain values, in the same
order and with the same branch precedence, every step annotated with the research line
it mirrors. The claim that it is the *same function* is tested, not asserted:
`tests/test_feature_equivalence.py` runs both paths over hundreds of randomised users —
including malformed ones and ones the pipeline legitimately refuses — and requires
identical brand order, identical ranks and identical payouts. `serving.feature_path:
research` switches back to the original at runtime.

### Cold start

`nd = NameDataset()` at module import in both research scripts costs **18.6 s and
2.4 GB resident**. Three worker processes would need 7 GB and 18 s each before serving
a single request.

`detect_gender_with_confidence` is a pure function of one first name, and the dataset's
first-name universe is finite (727,556 entries), so the training job materialises the
whole function into a table that ships inside the model bundle:

| | import cost | resident | per lookup |
|---|---|---|---|
| `NameDataset()` | 18.6 s | 2.4 GB | 77 µs |
| precomputed table | 3.9 s | 290 MB | 0.09 µs |

The table is keyed by what the serving path actually looks up. The research code does
`str(x).strip().capitalize()` before the lookup, and `capitalize()` lowercases
everything after the first letter, so "Anne-Marie" is asked for as "Anne-marie".
names-dataset normalises internally and still finds it; a table keyed on the dataset's
own spelling did not. 141,897 of the 727,556 names — every hyphenated and multi-word
first name among them — differ from their own capitalisation, and each one silently
returned `unknown`. Building each entry by asking the research function with the
capitalised key makes the table exact, because that is the same question serving asks.

A table built before this fix is indistinguishable from a correct one by inspection -
same row count, same columns, same size - so the parquet now carries a key-scheme
marker and loading an older one logs a warning naming what is wrong and that a retrain
rebuilds it. It warns rather than refuses: an old bundle still serves, and turning a
degraded feature into an outage would be the worse trade.

Names the table does not contain return `('unknown', 0.0)`, which is what the research
implementation returns for a name the dataset does not know.
`tests/test_gender_lut.py` checks both, through the serving key, and deliberately
samples the names that differ under capitalisation — the earlier version compared raw
dataset spellings on both sides, so it exercised a path production never takes and
stayed green while a fifth of the dataset missed.

The serving image therefore does not install `names-dataset` at all, and the research
predictor module is imported lazily so a default serving process never touches it.

### Concurrency

Handlers are `def`, not `async def`. The work is CPU-bound; inside a coroutine it
blocks the event loop and every other in-flight request queues behind it — and the
handler's own timer cannot see that wait, so the endpoint looks healthy while users
wait seconds.

Parallelism is by process (`--workers 3` on 4 cores), with each worker pinned to one
BLAS thread. `--no-access-log` matters more than it sounds: logging every request took
p99 from 14 ms to 225 ms at 120 rps in the measurements this design is based on.

---

## Load simulation

`loadtest/run_load.py` is an **open-loop** generator: arrivals are scheduled on a
Poisson process at a fixed target rate, independent of how the server is doing. A
closed-loop client (send, wait, send) systematically hides tail latency — when the
server stalls the client stalls with it and simply stops offering load, so the requests
that would have queued are never sent. That is coordinated omission, and it makes a
struggling service look fine.

Latency is measured from the moment a request was *due*, so queueing is included, and
the `X-Process-Time-Ms` header is reported alongside it — the gap between the two is
queueing, and it is the first thing to look at when p99 leaves p50 behind. Percentiles
are nearest-rank, so every number printed is a real observation.

```bash
make loadtest        # sweeps 25 -> 400 rps
```

Results and the capacity reading are in [`docs/load-test.md`](docs/load-test.md).

Caveat stated plainly: on a 4-core box the generator competes with the server. The
number worth trusting is the rate at which *achieved* rps falls behind *target*.

---

## Data

The real `bl_full_data.csv` lives in a private Drive folder. Drop it at
`data/raw/bl_full_data.csv` and everything downstream works unchanged — `make data` is
skipped whenever a file is already there.

Without it, `src/bl_ranking/data/generate.py` produces a schema-faithful substitute. It
is not filler: the brand acceptance rules and payout scales are a real generative model,
so the classifier and regressor have something to learn, and
`tests/test_generated_data.py` asserts that every survey band, every label path and
every edge case the research code handles actually occurs.

### The quality gate

`data/ingest.py` enforces, once at the boundary, five invariants the research code
assumes and a warehouse dump does not guarantee. Two of them are real bugs in the
pipeline as given:

- **`sub1`/`sub2`/`sub3`.** The research code does `fillna('Other')` then `.astype(str)`.
  One null anywhere in the column makes pandas read the whole column as float64, so
  training sees `'1815195.0'` while a JSON request produces `'1815195'`. Three of the
  fourteen categorical features would miss on **every single request**, permanently,
  with nothing in any log to say so.
- **`campaign_id`.** Real ids are around 1.2e17, past float64's 53-bit mantissa. One
  null demotes the column and `120227360861540306` silently becomes `...304` in training
  while serving sends the exact integer. Both are fixed by reading those columns as text
  and converting afterwards — once a float has been rounded nothing downstream can undo it.

Also enforced: `cellphone` must survive `.astype(int)`, survey columns must expose the
`.str` accessor, and `payout` must be numeric. Every repair is counted and logged to
MLflow, so a jump in repairs is visible instead of quietly changing a feature.

The serving request schema applies the same normalisation to the same fields, which is
what keeps the two sides consistent.

---

## Configuration

Everything is in `conf/config.yaml`, overridable by environment variables prefixed
`BL_` with `__` for nesting, and logged to MLflow in full on every run.

```bash
BL_MODEL__PAYOUT__BACKEND=tabpfn_client   # which payout model serves
BL_SERVING__WORKERS=4                     # worker processes
BL_SERVING__FEATURE_PATH=research         # use the original pipeline
BL_DATA__LOOKBACK_DAYS=60                 # rolling training window
MLFLOW_TRACKING_URI=http://localhost:5000
TABPFN_TOKEN=...                          # tabpfn_client only
```

---

## Tests

```bash
make test        # fast suite, no TabPFN, no names-dataset, no network
make test-all    # adds the equivalence and gender-table checks (slower)
```

The ones that carry weight:

| test | what it protects |
|---|---|
| `test_feature_equivalence.py` | the fast path *is* the research path, over randomised users |
| `test_gender_lut.py` | the precomputed table *is* `detect_gender_with_confidence` |
| `test_null_sub_ids_...` | the train/serve skew above stays fixed |
| `test_research_code_checksum_...` | the vendored scripts have not been edited |
| `test_local_and_databricks_schedules...` | the two schedules cannot drift |
| `test_serving_degrades_rather_than_failing_when_a_backend_cannot_load` | a TabPFN outage at model load is a degraded ranking, not a 503 |

---

## Known limitations

- **The surrogate is an approximation.** Its fidelity is measured and logged, never
  assumed, and one environment variable switches to an exact backend. But if the
  business requires TabPFN's exact numbers on the synchronous path, the honest answer
  is a GPU endpoint or an asynchronous design, not a faster CPU.
- **`split_by_time` anchors the test window on `max(session_dt)`**, not on midnight, so
  its "day 7" bucket is only the minutes between the last session and the same clock
  time the next day. The per-day report for day 7 is therefore always thin. Research
  behaviour, preserved, and worth knowing when reading the log.
- **The training job stages the Delta snapshot back to CSV** so `pd.read_csv` in the
  research code stays untouched. It costs seconds in a weekly batch job and it is
  arguably *more* faithful, since the research dtype inference ran on a CSV.
- **Timestamps are assumed to be naive UTC.** An offset-carrying request is converted to
  UTC and the offset dropped. If the funnel logs local time instead, one function in
  `serving/schemas.py` changes.
- **The ranking is evaluated per model, not per ranking.** MAE on payout and F1 on
  acceptance are what the research code reports; what actually earns money is whether
  the top slot is the best brand. See [`docs/design.md`](docs/design.md).

---

## Part 2

Implementation, the dilemmas, the decisions, how it behaves in production and what I
would do next:

- [`docs/BL_ranking_part2.pptx`](docs/BL_ranking_part2.pptx) — 15 slides, with speaker
  notes. The deliverable.
- [`docs/design.md`](docs/design.md) — the same material in long form, with the
  reasoning the slides compress.
- [`docs/load-test.md`](docs/load-test.md) — the latency method and the full sweep.
