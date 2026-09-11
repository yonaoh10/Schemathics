"""Does the surrogate change the ranking a user actually sees?

The fidelity numbers logged during training compare the student's *payout* predictions
against the teacher's. That is one step removed from what matters. The list on screen is
ordered by `expected_payout = P(lead) x payout`, and the classifier is the same in both
cases, so agreement on payout does not automatically mean agreement on the order.

This scores the same users through the whole endpoint twice - once with the surrogate,
once with the TabPFN teacher it was distilled from - and compares the rankings
themselves: the brand in position one, the top three as a set, and the full ordering.

    python scripts/compare_backends.py --users 200

Needs the TabPFN weights (`pip install -e '.[tabpfn]'`). It is slow by construction,
because the whole point is that the teacher is slow: roughly 0.46 s per user on CPU
at the shipped `n_estimators=4`.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import replace
from pathlib import Path

from bl_ranking.config import PayoutSettings, Settings, resolve
from bl_ranking.data import schema
from bl_ranking.models import bundle as bundle_files
from bl_ranking.models.payout import PayoutContext, create_backend
from bl_ranking.serving.ranker import WARMUP_USER, BrandRanker, InsufficientSurveyData


def random_users(count: int, seed: int = 4242) -> list[dict]:
    """Users spread across the survey space, not variations on one profile."""
    rng = random.Random(seed)
    users = []
    for i in range(count):
        user = copy.deepcopy(WARMUP_USER)
        user["credit_score"] = rng.choice([b for b, _ in schema.CREDIT_SCORE_BANDS])
        user["loan_amount"] = rng.choice([b for b, _ in schema.LOAN_AMOUNT_BANDS])
        user["monthly_revenue"] = rng.choice([b for b, _ in schema.MONTHLY_REVENUE_BANDS])
        user["time_in_business"] = rng.choice([b for b, _ in schema.TIME_IN_BUSINESS_BANDS[:4]])
        user["industry"] = rng.choice(schema.INDUSTRIES)
        user["business_type"] = rng.choice(schema.BUSINESS_TYPES)
        user["loan_reason"] = rng.choice(schema.LOAN_REASONS)
        user["device_type"] = rng.choice(schema.DEVICE_TYPES)
        user["page"] = rng.choice([p for p, _ in schema.PAGES])
        user["auto_state"] = rng.choice(["Florida", "Texas", "California", "New York"])
        user["auto_city"] = rng.choice(["Miami", "Austin", "Oakland", "Buffalo"])
        user["fname"] = rng.choice(["Michael", "Jennifer", "Maria", "Rigoberto", "Ahmed"])
        user["cellphone"] = 2010000000 + i
        day, hour = rng.randint(1, 28), rng.randint(0, 23)
        user["session_dt"] = f"2026-01-{day:02d} {hour:02d}:{rng.randint(0, 59):02d}:11"
        user["register_date"] = f"2026-01-{day:02d} {hour:02d}:{rng.randint(0, 59):02d}:44"
        users.append(user)
    return users


def latest_bundle(settings: Settings) -> Path:
    candidates = sorted(resolve(settings.paths.run_root).glob("*production*/bundle"))
    if not candidates:
        raise SystemExit("no production bundle found; run `make train-prod` first")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.load()
    bundle = args.bundle or latest_bundle(settings)
    manifest = bundle_files.Manifest.read(bundle)
    if manifest.payout_backend != "surrogate":
        raise SystemExit(
            f"bundle was built with {manifest.payout_backend!r}; this comparison only "
            f"means something for a surrogate bundle"
        )

    served = BrandRanker.load(bundle, settings)

    # The teacher, configured exactly as the distillation step configured it, so any
    # difference measured here is the student's and not a configuration drift.
    context = PayoutContext.from_artifact(bundle / bundle_files.PAYOUT_CONTEXT_FILE)
    teacher_cfg = PayoutSettings(backend=settings.model.payout.teacher,
                                 fit_mode=settings.model.payout.fit_mode,
                                 n_estimators=settings.model.payout.n_estimators,
                                 device=settings.model.payout.device)
    started = time.perf_counter()
    teacher = create_backend(teacher_cfg, teacher_cfg.backend).fit(context.x, context.y)
    print(f"teacher fitted in {time.perf_counter() - started:.1f}s "
          f"({teacher_cfg.backend}, {teacher_cfg.fit_mode}, "
          f"n_estimators={teacher_cfg.n_estimators})")

    reference = BrandRanker(replace(served.warm, payout=teacher),
                            feature_path=served.feature_path)

    users = random_users(args.users)
    top1 = top3 = full = scored = 0
    regrets: list[float] = []
    student_ms: list[float] = []
    teacher_ms: list[float] = []

    for i, user in enumerate(users, 1):
        try:
            t = time.perf_counter()
            a = served.rank(user)
            student_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            b = reference.rank(user)
            teacher_ms.append((time.perf_counter() - t) * 1000)
        except InsufficientSurveyData:
            continue
        if not a or not b:
            continue
        scored += 1
        order_a, order_b = list(a), list(b)
        top1 += order_a[0] == order_b[0]
        top3 += set(order_a[:3]) == set(order_b[:3])
        full += order_a == order_b

        # The number that decides whether the disagreements matter. Agreement counts
        # treat "picked a brand worth $0.02 less" the same as "picked a much worse
        # brand". Regret does not: it asks what the student's choice is worth *under
        # the teacher's own scores*, relative to the teacher's own pick.
        best = b[order_b[0]]["expected_payout"]
        chosen = b.get(order_a[0], {}).get("expected_payout", 0.0)
        if best > 0:
            regrets.append((best - chosen) / best)

        if i % 25 == 0:
            print(f"  {i}/{len(users)} ...", flush=True)

    student_ms.sort()
    teacher_ms.sort()
    regrets.sort()
    disagreements = [r for r in regrets if r > 0]
    result = {
        "users_scored": scored,
        "top1_agreement": round(top1 / scored, 4) if scored else None,
        "top3_set_agreement": round(top3 / scored, 4) if scored else None,
        "full_order_agreement": round(full / scored, 4) if scored else None,
        "mean_regret": round(sum(regrets) / len(regrets), 5) if regrets else None,
        "p95_regret": round(regrets[int(len(regrets) * 0.95)], 5) if regrets else None,
        "max_regret": round(regrets[-1], 5) if regrets else None,
        "mean_regret_when_disagreeing": (
            round(sum(disagreements) / len(disagreements), 5) if disagreements else 0.0),
        "student_p50_ms": round(student_ms[len(student_ms) // 2], 3) if student_ms else None,
        "teacher_p50_ms": round(teacher_ms[len(teacher_ms) // 2], 1) if teacher_ms else None,
        "bundle": str(bundle),
        "model_run_id": manifest.mlflow_run_id,
    }

    print()
    print(f"users scored               {result['users_scored']}")
    print(f"same brand in position 1   {result['top1_agreement']:.1%}")
    print(f"same top 3 (as a set)      {result['top3_set_agreement']:.1%}")
    print(f"identical full ordering    {result['full_order_agreement']:.1%}")
    print()
    print("expected-payout regret (student's pick, scored by the teacher):")
    print(f"  mean over all users      {result['mean_regret']:.2%}")
    print(f"  mean when they disagree  {result['mean_regret_when_disagreeing']:.2%}")
    print(f"  p95                      {result['p95_regret']:.2%}")
    print(f"  worst single user        {result['max_regret']:.2%}")
    print()
    print(f"surrogate p50              {result['student_p50_ms']:.2f} ms")
    print(f"teacher   p50              {result['teacher_p50_ms']:.1f} ms")
    speedup = result["teacher_p50_ms"] / result["student_p50_ms"]
    print(f"speed-up                   {speedup:,.0f}x")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
