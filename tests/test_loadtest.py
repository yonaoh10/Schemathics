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


def test_achieved_rate_is_measured_over_the_whole_span_not_the_offered_window():
    """A long drain has to push achieved below target, because the responses did not all
    arrive inside the offered window - dividing by that window credits the server with the
    drain. 500 responses whose last arrives 40 s after the 10 s window closed is 10 rps over
    the 50 s span, not the 50 rps a window-only denominator would report. `span_s` is set
    here on purpose: without it `summary()` falls back to `duration_s`, which makes both
    denominators 10 s and lets a divide-by-the-window regression pass unnoticed.
    """
    result = Result(target_rps=100, duration_s=10, offered=1000)
    result.span_s = 50.0
    result.samples = [_sample(due=i / 100, sent=i / 100, finished=50.0)
                      for i in range(500)]
    assert result.summary()["achieved_rps"] == 10.0


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


def test_a_payload_count_below_one_is_refused():
    """An empty synthetic set reached _drive, where payloads[i % len(payloads)] is a
    modulo by zero raised several arrivals in. Refused by name up front, like the empty
    --payloads file, rather than as a ZeroDivisionError mid-run."""
    with pytest.raises(SystemExit, match="payload-count must be at least 1"):
        load_payloads(None, count=0)


def test_the_table_rule_is_exactly_as_wide_as_the_header():
    """The rule was a fixed 116 dashes under a 105-character header. Tied to the header's
    own length now, so the two cannot drift when a column is added or widened."""
    lines = render([]).splitlines()
    header = next(ln for ln in lines if "verdict" in ln)
    rule = next(ln for ln in lines if set(ln) == {"-"})
    assert len(rule) == len(header)


def test_the_built_in_payloads_are_distinct_users():
    """One repeated user measures a perfectly cached code path.

    Distinctness has to hold across the fields the model reads, not just the cellphone:
    `cellphone` is a rolling integer, so 200 otherwise-identical users would pass a
    byte-string check while still being one cached scoring path. Dropping it and requiring
    variation in what remains is what makes this test able to fail.
    """
    payloads = load_payloads(None, count=50)
    assert len(payloads) == 50
    assert len(set(payloads)) == 50
    without_phone = set()
    for payload in payloads:
        user = json.loads(payload)
        user.pop("cellphone", None)
        without_phone.add(json.dumps(user, sort_keys=True))
    assert len(without_phone) > 1


def test_every_built_in_payload_is_one_the_endpoint_accepts():
    """The generator drew session_dt and register_date on independent days, so half its
    payloads had the survey submitted before the session that showed it - which the schema
    refuses, correctly. The sweep then measured a service answering 422s at half the offered
    rate and printed it as capacity, with clean percentiles of the refusals.

    Validated through the request model itself rather than by re-checking the date
    arithmetic here, so any rule the endpoint adds later is enforced on this workload too.
    All 200 are checked, not a prefix: 200 is `--payload-count`'s default and what the
    sweep actually sends, so checking 100 left the second half of the real workload unseen.
    """
    from bl_ranking.serving.schemas import RankRequest

    for payload in load_payloads(None, count=200):
        RankRequest(**json.loads(payload))


def test_a_run_that_mostly_errors_is_not_reported_as_latency():
    """The row said "ok" while half the traffic was a 422. Percentiles of refusals look
    excellent, so the verdict has to say what happened instead - and it is checked as the
    verdict `summary()` computed, not as a substring that also occurs in the table header.

    Built through Result so the verdict comes from the code under test rather than a hand
    dict: a hand dict can carry any verdict, which is how the old "  ok" check passed
    against the header even when no row was actually marked ok.
    """
    def summary(ok, errors, status=503):
        result = Result(target_rps=100, duration_s=10, offered=ok + errors)
        result.span_s = 10.0
        result.samples = [_sample(due=0.0, sent=0.0, finished=0.01) for _ in range(ok)]
        result.samples += [
            _sample(due=0.0, sent=0.0, finished=0.01, status=status, server_ms=None)
            for _ in range(errors)
        ]
        return result.summary()

    mostly_errors = summary(ok=500, errors=500)
    assert mostly_errors["verdict"] == "50% NOT 2xx"
    text = render([mostly_errors])
    assert "50% NOT 2xx" in text
    assert "not a latency measurement" in text

    # One reset in a thousand is not a story, and must not bury the real verdict.
    healthy = summary(ok=1000, errors=1)
    assert healthy["verdict"] == "ok"
    rendered = render([healthy])
    assert "NOT 2xx" not in rendered
    data_row = next(ln for ln in rendered.splitlines() if ln.strip().startswith("100"))
    assert data_row.rstrip().endswith("ok")


def test_a_run_with_no_responses_at_all_is_still_given_a_verdict():
    """A 100%-error row used to take the "no successful responses" branch and never get an
    error verdict - the one case the error check missed. A dead port (transport failures,
    no bodies) must read as "no response", not as the payload advice meant for non-2xx."""
    dead_port = Result(target_rps=100, duration_s=10, offered=200)
    dead_port.span_s = 10.0
    dead_port.errors = 200                    # connection failures: no samples at all
    summary = dead_port.summary()

    assert summary["verdict"] == "100% no response"
    assert summary["transport_errors"] == 200
    text = render([summary])
    assert "100% no response" in text
    assert "GET /readyz" in text
    # The payload advice is for non-2xx bodies, of which there were none here.
    assert "Fix the payloads" not in text
