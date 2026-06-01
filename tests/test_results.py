import pytest

from llama_benchy.client import RequestResult
from llama_benchy.results import BenchmarkResults


def test_decode_throughput_uses_observed_token_interval():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=1.0,
        first_token_ts=1.0,
        end_ts=1.35,
        prompt_tokens=100,
        total_tokens=4,
        token_timestamps=[1.0, 1.1, 1.2, 1.3],
    )

    results = BenchmarkResults()
    results.add("model", 100, 4, 0, 1, [[result]], latency=0.0, expected_pp_tokens=100)

    assert results.runs[0].tg_throughput is not None
    assert results.runs[0].tg_throughput.mean == pytest.approx(10.0)


def test_burst_output_does_not_report_decode_throughput():
    result = RequestResult(
        start_ts=0.0,
        first_response_ts=3.0,
        first_token_ts=3.0,
        end_ts=3.001,
        prompt_tokens=2048,
        total_tokens=130,
        token_timestamps=[3.0] * 130,
    )

    results = BenchmarkResults()
    results.add("model", 2048, 1024, 0, 1, [[result]], latency=0.0, expected_pp_tokens=2048)

    run = results.runs[0]
    assert run.tg_throughput is None
    assert run.peak_throughput is not None
    assert run.peak_throughput.mean == pytest.approx(130.0)

    rows = results._generate_rows()
    tg_row = next(row for row in rows if row["test_name"] == "tg1024")
    assert tg_row["t_s"] is None
    assert tg_row["peak_ts"] is not None
