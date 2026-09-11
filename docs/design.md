# Implementation, dilemmas, decisions

Part 2 of the assignment. What was built, what was genuinely difficult, what I decided
and why, how it behaves in production, and what I would do next.

---

## 1. The shape of the problem

Two research scripts fit and apply two models whose product orders lender brands for a
user who is waiting on a landing page. The brief says: productize them, do not change
their functions and logic, schedule the retrain weekly, make versions identifiable and
reversible, expose a synchronous endpoint, and minimise latency.

Those last two constraints fight the first one. Most of this document is about where
that fight is real and how I resolved it.

The engineering position I took throughout: **the research code defines what the answer
is; it does not define where the work happens or how often it is repeated.** Moving work
out of a request, or computing the same function a faster way and proving it is the same
function, is productization. Changing what the function returns is not.

To make that auditable rather than rhetorical, the two scripts are vendored
byte-for-byte, every training run records a SHA-256 of them, and a test fails if the
digest moves.

---

## 2. Dilemma 1 — "do not change the logic" versus a 60 ms feature pipeline

**What I found.** With the models already warm, one request through the research
pipeline costs 60 ms p50 on the production model, and almost none of it is arithmetic. It is pandas
per-operation overhead: a cross join, a DataFrame built from a dict, four
`to_datetime` calls, a `Series.apply` that constructs a `pd.Series` per row for the
gender feature, and about twenty `.loc[mask, col] = value` assignments across the four
band mappings. Roughly a hundred pandas calls to transform fifteen rows. The cost is per
*operation*, so it does not shrink with the data.

**The options.**

1. Ship 60 ms and call it done. Defensible on the letter of the brief, indefensible on
   "minimise latency as possible" — it is the entire budget for a page that is loading.
2. Edit the transforms. Fastest to write, and exactly what the brief forbids.
3. Write the same transformation a second way, and *prove* the two agree.

**What I did.** Option 3. `serving/fast_features.py` computes the same features over
plain Python values and numpy, in the same order, with the same branch precedence, every
step annotated with the research line it mirrors. Both paths stay available at runtime.

The proof is the point. `tests/test_feature_equivalence.py` runs both over hundreds of
randomised users — well-formed, malformed, and ones the pipeline legitimately refuses —
and requires identical brand order, identical ranks and identical expected payouts. It
caught two divergences I had not anticipated, both of which turned out to be real
findings about the original code (§5).

**Result: 60.3 ms to 2.41 ms**, a factor of 25, with the original one environment
variable away.

Two things I deliberately did not do. I did not "improve" the band mappings whose
overlapping `str.contains` branches look like bugs — `'$50,000 - $99,999'` matches the
`'9,999'` rule before the correct one overwrites it, and the result is right only
because of statement order. That is the model's behaviour and changing it would change
predictions. And I did not touch the `-99` sentinels that make
`ratio_loan_amount_revenue` negative; they are a feature the model was fitted on.

---

## 3. Dilemma 2 — TabPFN on a synchronous path

This is the assignment's real question, and the honest answer is uncomfortable.

**What the research code does.** TabPFN is in-context learning: `fit()` is not training,
it hands the model its 1000-row context. The predictor calls `load_models()` — and
therefore `fit()` — inside `predict_()`, on every request. With the hosted client that
means serialising the whole context to parquet, uploading it, and posting a fit, before
a single brand is scored. Then a second round trip to predict. Per user.

**The observation everything follows from:** that context is frozen between weekly
retrains. It changes only when Sunday's job changes it. So the fit belongs in the weekly
job, and a request should pay for its own fifteen rows and nothing else.

**Measured here** (4 CPU, 15 GB, context 1000x25, one request = 15 brand rows). The
three `fit_mode` rows hold `n_estimators=2` so that only the fit mode varies:

| `tabpfn_local` `fit_mode` | start-up | payout prediction | max diff vs the library default |
|---|---|---|---|
| `low_memory` | 0.2 s | 37.3 s | $4.03 |
| `fit_preprocessors` (library default) | 0.4 s | 8.69 s | reference |
| `fit_with_cache` | 14.9 s | **0.26 s** | **$0.000045** |

`fit_with_cache` is a 33x improvement for predictions that differ by 4.5e-5 on payouts
of $60-140 — the context is encoded once at start-up instead of on every call. It is the
largest exact-preserving win available and the research code leaves it on the table.
`low_memory` is worth noting for the opposite reason: it is both the slowest mode and
the only one that differs materially, so it is never the right choice here.

It does not rescue the request path. At the shipped teacher setting — `fit_with_cache`,
`n_estimators=4` — a complete request measures **457 ms p50** over 200 users, against
**2.41 ms** for the served path.

**The decision.** A ladder, not a single answer, because different deployments have
different budgets:

- **`surrogate` is the default.** TabPFN labels a large sample offline at training time
  — the real training rows crossed with the brand universe, which is exactly the shape
  of an inference request — and a CatBoost student is fitted to those labels. It is the
  only option that meets the stated latency requirement on CPU.
- **`tabpfn_client`** is the exact path. The library supports removing the per-request
  fit directly: `fit()` returns a server-side `fitted_train_set_id`, and an estimator
  whose `model_id_` is assigned can predict without ever fitting. The weekly job fits
  once, the id ships in the bundle, and a request costs one round trip carrying only its
  own rows.
- **`tabpfn_local`** is exact, private, and right for batch or GPU.
- **`catboost_fallback`** is availability: if TabPFN is unreachable the endpoint keeps
  answering rather than returning 503 while a user waits.

**What it costs, measured — and a lesson about which metric to trust.** The training
job logs the student against the teacher on held-out users: rank correlation 0.993,
dollar error 6.2%, the same top brand every time. I did not believe that last figure,
because the distillation sample is drawn from the payout context — rows that were
actually paid for — and it compares payout predictions rather than the ranking they
produce. So I measured the thing itself: 200 randomly generated users, scored through
the whole endpoint twice.

| | surrogate vs teacher |
|---|---|
| same brand in position 1 | 84.5% |
| same top 3, as a set | 82.0% |
| identical ordering of all 15 | 3.5% |
| **expected payout given up, mean over all users** | **1.21%** |
| ... when they disagree | 7.83% |
| ... worst single user | 27.8% |
| latency | 4.1 ms vs 456.7 ms |

The agreement rows are the wrong thing to read. They treat "picked a brand worth two
cents less" the same as "picked a much worse brand". The regret row asks what the
student's choice is worth *under the teacher's own scores*: the student disagrees about
first position for roughly one user in six, and when it does it gives up about 8% of
that user's expected payout — 1.21% averaged over everyone.

**So the trade is 1.21% of expected payout per session against 450 ms of added page
latency.** Which side wins depends on the funnel's latency-to-conversion curve, which
the business has and I do not. I defaulted to the surrogate because 450 ms on a landing
page is very likely to cost more than 1.2% of revenue — but the whole point of measuring
it this way is that someone with the conversion data can overturn the default with one
environment variable.

It also changed how I think about the logged metric. `surrogate_top1_agreement` at
training time is a regression detector, not an estimate of production behaviour; the
end-to-end script is the estimate. Both are kept, labelled for what they are.

**The uncomfortable part, stated plainly.** The surrogate changes predictions. I did not
want to hide that behind a config default, so: the distillation step measures itself
against the teacher (MAE, MAPE, Spearman) and logs the result to MLflow on every run;
every model version is tagged `payout_exact`; and every HTTP response carries
`payout_exact: false`. If the business needs TabPFN's exact numbers on the synchronous
path, the honest answer is a GPU endpoint or an asynchronous design — not a faster CPU.

Two smaller decisions with production consequences. The hosted checkpoint is **pinned**:
left at its default the provider chooses, so a server-side upgrade would change payouts
between two weekly runs with nothing in MLflow to explain the shift. And TabPFN sends
usage telemetry to a third party by default, which the serving image disables — relevant
beyond latency, because the hosted client uploads the in-context rows, and those are
real leads carrying name-derived and geographic features.

---

## 4. Dilemma 3 — the data is not available

The extract lives in a private Drive folder I could not reach. Two options: stub
something in and hope, or build a substitute good enough that the pipeline is genuinely
exercised.

I built the substitute, and made its fidelity testable. Brand acceptance is a logistic
function of a standardised user profile with per-brand weights, and payout is a
brand-specific scale times user quality — so the classifier and the regressor have real
structure to recover and the evaluation numbers mean something.
`tests/test_generated_data.py` asserts that every survey band maps to its documented
numeric value through the research code, that all four `client_buy` paths fire, that
both label classes appear in every test-day bucket, and that the malformed rows a real
extract always contains are present.

Dropping the real CSV at `data/raw/bl_full_data.csv` makes the generator step a no-op.
Nothing downstream changes.

Building the generator also turned out to be how I found the train/serve skew in §5:
writing a faithful generator forced me to work out exactly which columns can be null,
and that is precisely where the original pipeline breaks.

---

## 5. What the original code does wrong

Found by reading, then reproduced. Each is fixed at a boundary rather than inside the
protected code.

**A latent `NameError` in serving.** `bl_exp_payout_predictor.py:58` reads the module
global `user_data`, not `self.user_data`. Run as a script it works, because `__main__`
defines that global — and silently scores the hard-coded example for every caller.
Imported as a library, which is what serving does, it raises. The constructor argument
is otherwise unreachable. The fix is one token, and it is one of exactly three
deviations in `serving/research_path.py`, each marked inline: this, the cached brand
universe replacing a per-request `read_csv`, and two `logger.info` calls dropped to
`debug` so the endpoint does not emit two lines per request. Every feature expression
in that method is byte-for-byte the original.

**Attribution ids skew between training and serving.** `fillna('Other')` then
`.astype(str)` on `sub1`/`sub2`/`sub3`. One null anywhere in the column makes pandas read
it as float64, so training sees `'1815195.0'` while a JSON request produces `'1815195'`.
Three of the fourteen categorical features would miss on **every single request**,
permanently, with nothing in any log to say so. Fixed by reading those columns as text at
ingestion — once a float has been rounded nothing downstream can undo it.

**`campaign_id` loses precision.** Real ids are around 1.2e17, past float64's 53-bit
mantissa. One null demotes the column and `120227360861540306` becomes `...304` in
training while serving sends the exact integer. Same fix, same reason.

**Three input assumptions a warehouse dump does not guarantee.** `cellphone` must
survive `.astype(int)` (a real number like `(305) 555-0142` raises); survey columns must
expose `.str`; `payout` must be numeric. Enforced once at ingestion, with every repair
counted and logged so a jump is visible rather than silently changing a feature.

**A response shape that cannot happen.** The documented
`{"expected_payout": 0, "prob_lead": 0}` fallback is unreachable from the survey path.
When `dropna(thresh=5)` empties a single user's frame, execution continues into
`additional_features` and dies on `Columns must be same length as key`, because `.apply`
over an empty Series returns a Series rather than a two-column frame. The serving layer
checks survey completeness up front, identically on both feature paths, and returns 422.
My equivalence test found this: the two paths disagreed, and the research path was the
one that was wrong.

**Global state in library code.** `setup_bl_logger` clears the root logger's handlers
and opens a new timestamped file *per instantiation* — one log file per request, and it
silences MLflow and uvicorn for the rest of the process. `capture_warnings` rebinds a
global warning handler the same way. Both are hoisted to start-up; the training job
restores the root logger afterwards.

**A locale dependency.** `session_day_of_week` comes from `Series.dt.day_name()`, which
follows `LC_TIME`. A container with a different locale emits `'Dienstag'` — a categorical
level the model has never seen — with no error anywhere. `LC_ALL` is pinned in both
images.

---

## 6. How it runs in production

**Sizing.** The training job peaks around 9.5 GB resident on the full two-month window:
2.4 GB of it is the names-dataset import, the rest is the 81k-row frame and the
teacher-labelled distillation sample. That is what the cluster node type in
`databricks.yml` is sized for. Serving is a different shape entirely — roughly 400 MB
per worker, because the gender table replaced the library.

**Weekly, Sunday 05:00.** A Databricks job with three tasks: ingest the extract into a
Delta table, train on all data and register a version, then run the evaluation mode for
the researcher log and the comparable metrics. Evaluation runs last so a slow report
never delays the model reaching the endpoint. Locally the same three steps run from one
APScheduler process reading the same Quartz string.

Evaluation costs about as much as training, because it distils the same surrogate rather
than evaluating a cheaper proxy. That is deliberate: the numbers in the researcher log
should describe the model that is actually being served. It also explains the shape of
the job - registration first, evaluation second - so a slow report never sits between a
finished model and the endpoint.

**One run produces one version.** The research code writes four loose files, and loose
files are a rollback hazard: copy the classifier and the payout context separately and a
process can end up with a mismatched pair. `prediction_expected_payout` reorders the
frame by the *payout* model's column list and then feeds it to the *classifier*, so a
mismatch mis-slots features silently rather than failing. So a bundle — classifier,
payout context, brand universe, gender table, manifest — is registered atomically.

**Rollback is a pointer move.** `make rollback VERSION=3` moves the `champion` alias and
restarts the endpoint; the previous version is recorded under `champion_previous` so the
rollback is itself reversible. The serving endpoint follows the alias rather than a
pinned version, so a promotion is a restart, not a redeploy. `GET /model` reports what a
live worker actually loaded.

**Failure behaviour, by design.**

| failure | what happens |
|---|---|
| TabPFN unreachable when a worker loads its model | degrade to `catboost_fallback`, tag the response, keep answering |
| TabPFN lost *after* a worker started healthy | not covered: that worker 500s until restart. Wants a circuit breaker |
| TabPFN unreachable at train time | fail the run loudly; the current champion keeps serving |
| a retrain fails | the scheduler logs and survives; the alias does not move; next week retries |
| a worker cannot load its model | `/readyz` stays 503 so the load balancer skips it; `/healthz` stays green |
| sparse or malformed request | 422 with an empty ranking; the funnel falls back to its static order |
| unseen brand, campaign or city | scored normally — a new campaign must never be an outage |

**Warm-up is enforced, not hoped for.** `/readyz` only turns green after a real synthetic
scoring call has succeeded inside that worker, so the first user never pays the model
load. The Databricks endpoint is configured `scale_to_zero_enabled: false` for the same
reason: a cold start on a landing page is a lost session.

**What is monitored.** Request rate, latency histogram and error rate by outcome via
Prometheus; per-run ingest repair counters and the share of rows landing on the `-99`
sentinel for each band mapping. That last one matters more than it looks: a copy change
that renames a funnel answer sends every user to `-99` silently, and it would show up as
a metric before it showed up as revenue.

---

## 7. Alternatives I considered and rejected

**Rewrite the feature engineering once and share it between training and serving.** The
two scripts are near-duplicate code, and a single shared implementation would remove a
whole class of skew. Rejected: it is the change the brief explicitly forbids. The
equivalence test gets most of the benefit — it *proves* the two agree — without the edit.

**Keep the models loaded but leave the pipeline alone.** This was the first version,
and it is worth roughly a fifth of the total: hoisting start-up work removes the
per-request model loading, but the ~60 ms of pandas overhead in the feature pipeline is
untouched by it. Good, and nowhere near enough for a page load.

**Reduce TabPFN's `n_estimators`.** Cheap and tempting. Measured, it reorders brands —
Spearman against the default is below 1.0 — and brand order is the output that matters.
Rejected as a silent knob; it stays configurable and documented.

**Keep `names-dataset` and share it across workers with a preloading fork.** Would cut
the 2.4 GB from N copies to roughly one. Rejected: it is a workaround for a cost that can
be removed entirely. Materialising the gender function into a 4 MB table makes the
serving image not need the library at all, and it is exact by construction because the
name universe is finite.

**Serve the MLflow pyfunc directly instead of FastAPI.** Simpler, and the pyfunc is
registered with every model version, so `mlflow models serve -m models:/bl_brand_ranker@champion`
works and the Databricks endpoint uses exactly that artifact. Rejected as the primary
local path: the funnel wants a plain JSON dictionary, not `{"dataframe_records": [...]}`,
and readiness, metrics and a typed request schema are not optional on a revenue path.

**A closed-loop load test.** Every worker sends, waits, sends. Rejected: it hides tail
latency exactly when a service is struggling, because a stalled server stalls the client
and the requests that would have queued are never sent.

---

## 8. What I would do next

**Measure the thing that earns money.** Every metric here is per model — F1 on
acceptance, MAE on payout. What actually matters is whether the *top slot* is the best
brand. A replay evaluation over historical sessions, scoring realised payout of the
ranking's first position against the best achievable, would give a number the business
recognises, and would let the surrogate be judged on rank agreement rather than on
dollar error.

**Close the loop on the surrogate.** Log a sample of production requests, score them
with the teacher offline, and alert when student-teacher divergence or top-1 disagreement
drifts. The distillation fidelity is measured at training time today; it should be
monitored continuously.

**Challenger traffic.** The alias mechanism already supports `champion` and
`challenger`. A small traffic split with realised-payout comparison turns the weekly
retrain from "hope it is better" into a decision.

**Feature-contract enforcement at startup.** The bundle knows which columns are
categorical and what dtypes they had at fit time. A worker should assert its own frame
reproduces them before reporting ready, which would have caught the `sub1` skew
automatically instead of by reading.

**Drift monitoring on the inputs.** Databricks inference tables are already enabled in
the bundle. The signal to watch is the band-sentinel share and the categorical novelty
rate — those move first when the funnel changes.

**A circuit breaker around the hosted payout backend.** Today the fallback to
`catboost_fallback` happens when a worker cannot construct its backend at load time.
A worker that starts healthy and then loses the hosted API returns 500s until it is
restarted. The right shape is a breaker that trips after N consecutive failures,
serves the fallback while open, and retries on a timer — not a silent per-request
model swap, which would make the ranking quietly non-deterministic under partial
outage.
