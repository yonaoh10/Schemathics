"""The load generator's own arithmetic.

The harness is the instrument the latency claims are made with, and an instrument that
reports its own saturation as the server's is worse than no instrument: it produced a
report saying 54 s p50 at 400 rps while an independent client, asking the same server at
the same moment, got 8.3 ms. Everything here is about the numbers rather than about
traffic, so no server is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loadtest"))

from run_load import (  # noqa: E402
    Result,
    Sample,
    _percentiles,
    _target_rates,
    load_payloads,
    render,
)


def _sample(due: float, sent: float, finished: float, status: int = 200,
            server_ms: float | None = 5.0) -> Sample:
    return Sample(due=due, sent=sent, finished=finished, status=status, server_ms=server_ms)


def test_percentiles_are_nearest_rank_and_every_value_is_real():
    """No interpolation, so a reported number is always something that happened."""
    values = [float(n) for n in range(1, 101)]
    result = _percentiles(values)
    assert result["p50"] == 50.0
    assert result["p90"] == 90.0
    assert result["p99"] == 99.0
    assert result["max"] == 100.0
    assert _percentiles([]) == {}
    # One observation: every quantile is that observation, not an average of nothing.
    assert _percentiles([7.0])["p999"] == 7.0


def test_a_late_generator_is_reported_as_the_generator():
    """The defect this file exists for. Requests that went out 5 s after they were due
    have a 5 s client latency and a 3 ms service latency, and the row used to be published
    as the service's p50."""
    result = Result(target_rps=400, duration_s=10, max_send_lag_ms=100.0, offered=2)
    result.samples = [_sample(due=0.0, sent=5.0, finished=5.003),
                      _sample(due=1.0, sent=6.0, finished=6.003)]
    summary = result.summary()

    assert summary["generator_saturated"] is True
    assert summary["send_lag_ms"]["p50"] == pytest.approx(5000, abs=1)
    assert summary["service_ms"]["p50"] == pytest.approx(3, abs=1)
    assert summary["client_ms"]["p50"] == pytest.approx(5003, abs=1)


def test_a_generator_that_keeps_up_is_not_flagged():
    result = Result(target_rps=50, duration_s=10, max_send_lag_ms=100.0, offered=2)
    result.samples = [_sample(due=0.0, sent=0.001, finished=0.009),
                      _sample(due=1.0, sent=1.002, finished=1.010)]
    summary = result.summary()

    assert summary["generator_saturated"] is False
    assert summary["send_lag_ms"]["p99"] < 100
    assert summary["client_ms"]["p50"] == pytest.approx(9, abs=1)


def test_achieved_rate_is_measured_over_the_offered_window():
    """Dividing by the observed span let a long drain push the figure below target and a
    late first arrival push it above - and that figure is what the capacity claim rests on.
    """
    result = Result(target_rps=100, duration_s=10, offered=1000)
    # 500 responses, the last of them arriving 40 s after the 10 s window closed.
    result.samples = [_sample(due=i / 100, sent=i / 100, finished=50.0)
                      for i in range(500)]
    assert result.summary()["achieved_rps"] == 50.0


def test_a_non_200_is_an_error_and_not_a_latency():
    result = Result(target_rps=10, duration_s=1, offered=2)
    result.samples = [_sample(due=0.0, sent=0.0, finished=0.01),
                      _sample(due=0.1, sent=0.1, finished=0.11, status=503, server_ms=None)]
    summary = result.summary()

    assert summary["ok"] == 1
    assert summary["errors"] == 1
    assert len(summary["client_ms"]) > 0
    # The failed request's latency is meaningless, but its send lag is not.
    assert summary["send_lag_ms"]["max"] >= 0


@pytest.mark.parametrize("text", ["-5", "0", "abc", "", "nan", "inf", "5,-1"])
def test_a_rate_that_would_break_the_scheduler_is_refused(text):
    """A negative rate gave expovariate a negative mean, so every arrival was due before
    the last one and the loop allocated tasks as fast as it could: 4.5 GB resident in 51
    seconds, from a flag typo."""
    with pytest.raises(SystemExit):
        _target_rates(text)


def test_valid_rates_still_parse():
    assert _target_rates("25,50,100.5") == [25.0, 50.0, 100.5]


def test_a_payload_file_that_is_not_there_is_refused(tmp_path):
    """It used to fall through to the synthetic payloads, so a typo produced a healthy
    report for a workload nobody asked about."""
    with pytest.raises(SystemExit, match="does not exist"):
        load_payloads(tmp_path / "nope.json", count=10)

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(SystemExit, match="no records"):
        load_payloads(empty, count=10)


def test_the_built_in_payloads_are_distinct_users():
    """One repeated user measures a perfectly cached code path."""
    payloads = load_payloads(None, count=50)
    assert len(payloads) == 50
    assert len(set(payloads)) == 50


def test_every_built_in_payload_is_one_the_endpoint_accepts():
    """The generator drew session_dt and register_date on independent days, so half its
    payloads had the survey submitted before the session that showed it - which the schema
    refuses, correctly. The sweep then measured a service answering 422s at half the offered
    rate and printed it as capacity, with clean percentiles of the refusals.

    Validated through the request model itself rather than by re-checking the date
    arithmetic here, so any rule the endpoint adds later is enforced on this workload too.
    """
    from bl_ranking.serving.schemas import RankRequest

    for payload in load_payloads(None, count=100):
        RankRequest(**json.loads(payload))


def test_a_run_that_mostly_errors_is_not_reported_as_latency():
    """The row said "ok" while half the traffic was a 422. Percentiles of refusals look
    excellent, so the verdict column has to say what happened instead."""
    def summary(ok, errors):
        return {"target_rps": 100, "achieved_rps": 50.0, "ok": ok, "errors": errors,
                "service_ms": {"p50": 1.0, "p99": 2.0, "max": 3.0},
                "send_lag_ms": {"p50": 0.1, "p99": 0.2},
                "client_ms": {"p50": 1.1, "p99": 2.2},
                "server_ms": {"p50": 0.9, "p99": 1.8},
                "generator_saturated": False, "saturation_reasons": [],
                "offered": ok + errors, "max_in_flight": 4, "processes": 3}

    mostly_errors = render([summary(ok=500, errors=500)])
    assert "50% NOT 2xx" in mostly_errors
    assert "not a latency measurement" in mostly_errors

    # One reset in a thousand is not a story, and must not bury the real verdict.
    healthy = render([summary(ok=1000, errors=1)])
    assert "NOT 2xx" not in healthy
    assert "  ok" in healthy
