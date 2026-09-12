"""Where a single /rank request spends its time, stage by stage.

docs/load-test.md publishes a table of per-stage costs and the research path's figure to
compare against. Both used to be measured by hand, which meant nobody could check them
against the model actually in the bundle. This reproduces them:

    python scripts/profile_request.py                  # 200 distinct users
    python scripts/profile_request.py --users 1        # one user, repeated: cache-friendly
    python scripts/profile_request.py --no-research    # skip the 2.4 GB names-dataset import

The stages are the same calls `BrandRanker._rank_fast` makes, timed individually and then
compared against one timed `rank()` so the parts are known to add up. The research figure
runs the unmodified pipeline through `serving.feature_path: research` with the models
already warm, which flatters it: the original also loads a model per request.

Numbers move with the machine and with the model in the bundle. Quote the run, not the
file: the header line says which bundle and how many rows it was fitted on.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

# A sibling script imported as a module: `python scripts/profile_request.py` puts scripts/
# on the path. The user generator lives there and is shared rather than copied, so the two
# scripts cannot drift into profiling different traffic.
from compare_backends import random_users  # noqa: I001

from bl_ranking.config import Settings
from bl_ranking.serving import batch, fast_features
from bl_ranking.serving.model_source import resolve_bundle
from bl_ranking.serving.ranker import BrandRanker


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank, like the load generator: every number printed is an observation."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def profile(ranker: BrandRanker, users: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Time each stage of the fast path, for every user, in the order the request runs."""
    stages: dict[str, list[float]] = {name: [] for name in (
        "features", "broadcast", "pool", "classifier", "payout", "rank", "total")}
    cat_indices = ranker._cat_indices
    warm = ranker.warm
    for user in users:
        started = time.perf_counter()
        row = fast_features.build_feature_row(user, warm.gender)
        after_features = time.perf_counter()
        scoring = batch.from_row(row, warm.columns, ranker._brands)
        after_broadcast = time.perf_counter()
        pool = scoring.pool(cat_indices)
        after_pool = time.perf_counter()
        prob_lead = warm.catboost.predict_proba(pool)[:, 1]
        after_classifier = time.perf_counter()
        payout = warm.payout.predict_batch(scoring)
        after_payout = time.perf_counter()
        fast_features.rank_from_scores(ranker._brands, prob_lead, payout)
        finished = time.perf_counter()

        stages["features"].append(after_features - started)
        stages["broadcast"].append(after_broadcast - after_features)
        stages["pool"].append(after_pool - after_broadcast)
        stages["classifier"].append(after_classifier - after_pool)
        stages["payout"].append(after_payout - after_classifier)
        stages["rank"].append(finished - after_payout)
        stages["total"].append(finished - started)
    return stages


def whole_calls(ranker: BrandRanker, users: list[dict[str, Any]]) -> list[float]:
    """`rank()` end to end, as the endpoint calls it. The check on the sum of the parts."""
    timings = []
    for user in users:
        started = time.perf_counter()
        ranker.rank(user)
        timings.append(time.perf_counter() - started)
    return timings


LABELS = {
    "features": "build the user's features (fast_features.build_feature_row)",
    "broadcast": "broadcast across the brands (batch.from_row)",
    "pool": "CatBoost Pool construction, shared by both models",
    "classifier": "classifier predict_proba",
    "payout": "payout predict",
    "rank": "sort, rank, serialise",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=200,
                        help="distinct users to score. 1 repeats a single user, which is "
                             "the friendliest possible case for cache locality")
    parser.add_argument("--repeat", type=int, default=1,
                        help="passes over the user list, after one discarded warm-up pass")
    parser.add_argument("--no-research", action="store_true",
                        help="skip the research-path comparison (imports names-dataset: "
                             "18 s and 2.4 GB)")
    args = parser.parse_args()

    settings = Settings.load()
    bundle = resolve_bundle(settings)
    ranker = BrandRanker.load(bundle, settings)
    described = ranker.describe()
    users = random_users(args.users) * args.repeat

    print(f"bundle        {bundle}")
    print(f"model version {described['model_version']}")
    print(f"payout        {described['payout_backend']} (exact={described['payout_exact']})")
    print(f"brands        {described['n_brands']}")
    print(f"users         {args.users} distinct, {len(users)} calls\n")

    profile(ranker, users[: min(len(users), 20)])          # warm-up, discarded
    stages = profile(ranker, users)
    def row(label: str, values: list[float]) -> None:
        print(f"{label:<52}" + "".join(f"{_percentile(values, q) * 1000:9.3f}"
                                       for q in (0.50, 0.95, 0.99)))

    print(f"{'stage':<52}{'p50':>9}{'p95':>9}{'p99':>9}")
    for name, label in LABELS.items():
        row(label, stages[name])
    total = stages["total"]
    row("total in-process (sum of the stages above)", total)

    whole = whole_calls(ranker, users)
    row("the same work through ranker.rank()", whole)
    print(f"\nmean of totals {statistics.mean(total) * 1000:.3f} ms, "
          f"mean of rank() {statistics.mean(whole) * 1000:.3f} ms")

    if args.no_research:
        return
    research_settings = Settings.load()
    research_settings.serving.feature_path = "research"
    reference = BrandRanker.load(bundle, research_settings)
    # Warm: the research path's first call pays imports the fast path never makes.
    reference.rank(users[0])
    research = whole_calls(reference, users)
    print(f"\nthe unmodified research pipeline, models already warm: "
          f"p50 {_percentile(research, 0.50) * 1000:.1f} ms, "
          f"p95 {_percentile(research, 0.95) * 1000:.1f} ms, "
          f"p99 {_percentile(research, 0.99) * 1000:.1f} ms")
    print(f"  {_percentile(research, 0.50) / _percentile(whole, 0.50):.0f}x the served path, "
          f"and it would also load a model per request")


if __name__ == "__main__":
    main()
