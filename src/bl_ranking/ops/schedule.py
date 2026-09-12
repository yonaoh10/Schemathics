"""The weekly production retrain: Sunday 05:00.

There is one schedule definition, in conf/config.yaml, written as a Quartz expression
because that is what the Databricks Jobs API requires:

    quartz_cron_expression: "0 0 5 ? * SUN *"
                             │ │ │ │ │  │  └── year
                             │ │ │ │ │  └───── day-of-week (SUN)
                             │ │ │ │ └──────── month
                             │ │ │ └────────── day-of-month ('?' - unset, because
                             │ │ │              day-of-week is the one in use)
                             │ │ └──────────── hour
                             │ └────────────── minute
                             └──────────────── second

Quartz is 6 or 7 fields and starts at seconds, unlike the 5-field unix cron. Getting
that wrong is the classic way to schedule a job for the wrong time, so the local runner
parses the same string instead of keeping a second copy in a different dialect.

Locally, `python -m bl_ranking.ops.schedule` runs an APScheduler process that fires
`training.job.run(train_test=False)` on that schedule. On Databricks the same expression
goes into the job definition in databricks.yml at the repository root, and no scheduler
process exists at all.

"Both are driven by `schedule.cron`" was too strong: a bundle cannot read our config file,
so the expression is copied into it by hand. What keeps them together is a test that reads
both files and fails when either moves - which is the only mechanism available, and worth
naming rather than implying a shared read that does not exist.

Two Quartz features are refused here rather than translated, because APScheduler cannot
express them and Databricks can: the calendar tokens L, W and # in day-of-month, and a
restricted year. Silently dropping either meant the two schedulers fired on different
days from the same string, which is the one outcome this module exists to prevent.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import UTC

from bl_ranking.config import Settings

log = logging.getLogger("bl_ranking.schedule")

_DAY_ALIASES = {
    "SUN": "sun", "MON": "mon", "TUE": "tue", "WED": "wed",
    "THU": "thu", "FRI": "fri", "SAT": "sat",
}


@dataclass(frozen=True)
class CronFields:
    """A Quartz expression decomposed into the fields APScheduler wants."""

    second: str
    minute: str
    hour: str
    day: str
    month: str
    day_of_week: str

    def as_apscheduler_kwargs(self) -> dict[str, str]:
        return {
            "second": self.second,
            "minute": self.minute,
            "hour": self.hour,
            "day": self.day,
            "month": self.month,
            "day_of_week": self.day_of_week,
        }


def parse_quartz(expression: str) -> CronFields:
    """Translate a Quartz cron into APScheduler fields.

    Only the parts this project uses are supported: numeric fields, `*`, `?`, and
    three-letter day names. Anything more exotic should be rejected loudly rather than
    silently scheduled at the wrong time.
    """
    parts = expression.split()
    if len(parts) not in (6, 7):
        raise ValueError(
            f"{expression!r} is not a Quartz expression: expected 6 or 7 fields "
            f"(second minute hour day-of-month month day-of-week [year]), got {len(parts)}"
        )
    second, minute, hour, day, month, day_of_week = parts[:6]

    # APScheduler has a `year` field; Quartz's seventh field was simply dropped, so
    # '0 0 5 ? * SUN 2030' fired every Sunday here and never on Databricks - the widest
    # possible disagreement between the two schedulers, from a field that was ignored.
    year = parts[6] if len(parts) == 7 else "*"
    if year.strip() not in {"*", "?", ""}:
        raise ValueError(
            f"{expression!r} restricts the year to {year!r}. That field was being dropped, "
            f"so the local runner fired on every matching day while Databricks fired only "
            f"inside that year - use '*' and let the schedule be turned off instead."
        )

    # Quartz requires exactly one of day-of-month / day-of-week to be '?'.
    # APScheduler has no '?', so it becomes '*'.
    day = "*" if day == "?" else day

    # Quartz's calendar tokens - L (last), W (nearest weekday), # (nth weekday of the
    # month) - have no APScheduler equivalent. Passed through, APScheduler answered
    # `Unrecognized expression "15W" for field "day"`, which names neither the cron nor
    # the setting it came from, and only when a worker started. Said here instead,
    # because the failure mode that matters is a schedule Databricks accepts and the
    # local runner cannot reproduce: the two would then disagree about when the weekly
    # retrain happens, which is the one thing this translation exists to prevent.
    unsupported = sorted({token for token in "LW#" if token in day.upper()})
    if unsupported:
        raise ValueError(
            f"{expression!r} uses Quartz calendar token(s) {', '.join(unsupported)} in "
            f"its day-of-month field ({day!r}). Databricks accepts those and the local "
            f"APScheduler runner cannot express them, so the two would disagree about "
            f"when the retrain runs. Use an explicit day, or schedule by day-of-week."
        )
    day_of_week = "*" if day_of_week == "?" else _translate_day_of_week(day_of_week)
    return CronFields(second=second, minute=minute, hour=hour, day=day,
                      month=month, day_of_week=day_of_week)


# Quartz numbers days 1=SUN..7=SAT; APScheduler numbers 0=MON..6=SUN. Passing a number
# through unchanged therefore moves the schedule by two days without any error - a
# weekly retrain configured as '1' for Sunday would have run on Tuesday. Both systems
# accept the same three-letter names, so numbers are translated into names and the
# ambiguity disappears.
_QUARTZ_DAY_NUMBERS = {1: "sun", 2: "mon", 3: "tue", 4: "wed", 5: "thu", 6: "fri", 7: "sat"}
_QUARTZ_DAY_BY_NAME = {name: number for number, name in _QUARTZ_DAY_NUMBERS.items()}

# APScheduler's own order, which is where the two systems disagree: Quartz starts the
# week on Sunday, APScheduler ends it there. So a range's *edges* cannot simply be
# translated - the set of days has to be translated and then re-expressed.
_APSCHEDULER_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _translate_day_of_week(field: str) -> str:
    """Render a Quartz day-of-week in a form APScheduler reads identically."""
    text = field.strip()
    if text in {"*", ""}:
        return "*"

    def one(token: str) -> str:
        token = token.strip()
        if token.isdigit():
            number = int(token)
            if number not in _QUARTZ_DAY_NUMBERS:
                raise ValueError(
                    f"{field!r} is not a Quartz day-of-week: {number} is outside 1-7 "
                    f"(1=SUN .. 7=SAT)"
                )
            return _QUARTZ_DAY_NUMBERS[number]
        name = _DAY_ALIASES.get(token.upper(), token.lower())
        if name not in set(_QUARTZ_DAY_NUMBERS.values()):
            raise ValueError(f"{field!r} is not a Quartz day-of-week: {token!r}")
        return name

    def days_in(part: str) -> list[str]:
        """The days one comma-separated part selects, as names.

        A Quartz range runs in Quartz's week, so it may wrap: '6-2' is FRI,SAT,SUN,MON.
        Translating the two edges and handing 'fri-mon' to APScheduler gets "The minimum
        value in a range must not be higher than the maximum" - which names neither the
        cron expression nor the setting it came from. Worse, because Sunday is 1 in
        Quartz and last in APScheduler, *every* range starting on Sunday wrapped: the
        ordinary Quartz weekday range '1-5' took the local runner down on start-up
        while Databricks accepted it happily. So walk the range in Quartz's week and
        collect the days themselves.
        """
        edges = part.split("-")
        if len(edges) == 1:
            return [one(edges[0])]
        if len(edges) != 2:
            raise ValueError(f"{field!r} is not a Quartz day-of-week: {part!r}")
        start, end = (_QUARTZ_DAY_BY_NAME[one(edge)] for edge in edges)
        walked, day = [], start
        while True:
            walked.append(_QUARTZ_DAY_NUMBERS[day])
            if day == end:
                return walked
            day = day % 7 + 1

    selected = {name for part in text.split(",") for name in days_in(part)}
    ordered = [name for name in _APSCHEDULER_DAY_ORDER if name in selected]

    # Kept as a range when the days happen to be contiguous in APScheduler's week, both
    # because it is what an operator wrote and because it reads back as one thing.
    # Otherwise an explicit list, which has no ordering to get wrong.
    first = _APSCHEDULER_DAY_ORDER.index(ordered[0])
    last = _APSCHEDULER_DAY_ORDER.index(ordered[-1])
    if len(ordered) > 1 and last - first + 1 == len(ordered):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(ordered)


def run_scheduler(settings: Settings | None = None, run_now: bool = False) -> None:
    """Block, running the production training job on the configured schedule."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    settings = settings or Settings.load()
    if settings.schedule.pause_status.upper() == "PAUSED":
        log.warning("schedule.pause_status is PAUSED; nothing will be scheduled")
        return

    fields = parse_quartz(settings.schedule.cron)
    trigger = CronTrigger(timezone=settings.schedule.timezone,
                          **fields.as_apscheduler_kwargs())

    scheduler = BlockingScheduler(timezone=settings.schedule.timezone)
    scheduler.add_job(
        _train,
        trigger=trigger,
        id="bl_weekly_production_train",
        name="BL weekly production retrain",
        # A retrain takes minutes; if one is still running when the next fires,
        # skipping is correct - two concurrent runs would race on the registry alias.
        max_instances=1,
        coalesce=True,
        # Tolerate a container restart around the trigger time rather than skipping
        # the week's retrain entirely.
        misfire_grace_time=3600,
    )

    log.info("scheduled %s (%s) -> next run %s", settings.schedule.cron,
             settings.schedule.timezone, trigger.get_next_fire_time(None, _now()))

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: scheduler.shutdown(wait=False))

    if run_now:
        log.info("--run-now given: training immediately, then waiting for the schedule")
        _train()
    scheduler.start()


def _train() -> None:
    """The weekly job: the same three steps, in the same order, as the Databricks job.

    All three, because this ran only the middle one. The Databricks job ingests, trains and
    then evaluates; this scheduler called `run(train_test=False)` and nothing else - so the
    documented local analogue retrained on whichever Delta version happened to be current
    when the container started, week after week, and the researcher log and the comparable
    metrics were never produced at all. A frozen input is the worst version of that: every
    weekly run reports a new model version trained on data that has not moved.

    Ingest failing stops the week: training on a stale snapshot while believing it was
    fresh is what this exists to prevent. Evaluation failing does not, because the model is
    already registered by then and its numbers are a report rather than a gate - which is
    also why it runs last here and last on Databricks.
    """
    from bl_ranking.data.ingest import ingest
    from bl_ranking.training.job import run

    started = time.perf_counter()
    log.info("weekly retrain starting: ingest, train, evaluate")
    try:
        report = ingest()
        log.info("ingested %s rows -> %s rows, delta version %s",
                 f"{report.rows_in:,}", f"{report.rows_out:,}", report.delta_version)
        result = run(train_test=False)
        log.info("retrain finished in %.1fs: run=%s version=%s",
                 time.perf_counter() - started, result.run_id, result.model_version)
    except Exception:
        # A failed retrain must not kill the scheduler: the currently aliased version
        # keeps serving, and the next window gets another attempt.
        log.exception("weekly retrain failed; the champion alias is unchanged")
        return

    try:
        evaluation = run(train_test=True)
        log.info("evaluation finished: run=%s", evaluation.run_id)
    except Exception:
        # Deliberately not fatal, and deliberately after registration: the version is
        # already serving, and a missing report is not a reason to hold it back.
        log.exception("weekly evaluation failed; the registered version is unaffected")


def _now():
    from datetime import datetime
    return datetime.now(UTC)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Weekly retrain scheduler")
    parser.add_argument("--run-now", action="store_true",
                        help="train once on start-up, then follow the schedule")
    parser.add_argument("--show", action="store_true",
                        help="print the parsed schedule and exit")
    args = parser.parse_args()

    settings = Settings.load()
    if args.show:
        fields = parse_quartz(settings.schedule.cron)
        print(f"quartz       {settings.schedule.cron}")
        print(f"timezone     {settings.schedule.timezone}")
        print(f"apscheduler  {fields.as_apscheduler_kwargs()}")
        return
    run_scheduler(settings, run_now=args.run_now)


if __name__ == "__main__":
    main()
