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
goes into the job definition in databricks/databricks.yml and no scheduler process
exists at all. Both are driven by `schedule.cron`, so they cannot drift apart.
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

    # Quartz requires exactly one of day-of-month / day-of-week to be '?'.
    # APScheduler has no '?', so it becomes '*'.
    day = "*" if day == "?" else day
    day_of_week = "*" if day_of_week == "?" else _DAY_ALIASES.get(day_of_week.upper(), day_of_week.lower())
    return CronFields(second=second, minute=minute, hour=hour, day=day,
                      month=month, day_of_week=day_of_week)


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
    from bl_ranking.training.job import run

    started = time.perf_counter()
    log.info("weekly production retrain starting")
    try:
        result = run(train_test=False)
        log.info("retrain finished in %.1fs: run=%s version=%s",
                 time.perf_counter() - started, result.run_id, result.model_version)
    except Exception:
        # A failed retrain must not kill the scheduler: the currently aliased version
        # keeps serving, and the next window gets another attempt.
        log.exception("weekly retrain failed; the champion alias is unchanged")


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
