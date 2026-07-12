import csv
import io
import sys

from llama_benchy.config import BenchmarkConfig
from llama_benchy.results import BenchmarkMetric, BenchmarkResults, BenchmarkRun


def metric(value: float) -> BenchmarkMetric:
    return BenchmarkMetric(mean=value, std=0.0, values=[value])


def make_results() -> BenchmarkResults:
    results = BenchmarkResults()
    results.model_name = "test-model"

    for concurrency, multiplier in ((1, 1.0), (2, 1.8)):
        for depth in (2048, 4096):
            results.runs.append(
                BenchmarkRun(
                    concurrency=concurrency,
                    context_size=depth,
                    prompt_size=2048,
                    response_size=128,
                    is_context_prefill_phase=False,
                    pp_throughput=metric(1000.0 * multiplier - depth / 100),
                    pp_req_throughput=metric((1000.0 * multiplier - depth / 100) / concurrency),
                    tg_throughput=metric(50.0 * multiplier),
                    tg_req_throughput=metric(50.0 * multiplier / concurrency),
                    peak_throughput=metric(55.0 * multiplier),
                    peak_req_throughput=metric(55.0 * multiplier / concurrency),
                    ttfr=metric(100.0),
                    est_ppt=metric(90.0),
                    e2e_ttft=metric(120.0),
                )
            )

    return results


def test_sweep_defaults_expand_depths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llama-benchy",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
            "--sweep",
            "--sweep-max",
            "4096",
        ],
    )

    config = BenchmarkConfig.from_args()

    assert config.depths == [2048, 4096]
    assert config.sweep is True


def test_sweep_csv_contains_ds4_columns_and_concurrency():
    csv_output = make_results()._generate_sweep_csv()

    assert csv_output.startswith("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes")
    assert ",concurrency," in csv_output
    assert "2048,2048," in csv_output
    assert ",2," in csv_output


def test_sweep_svg_contains_concurrency_series():
    svg_output = make_results()._generate_sweep_svg("Test sweep")

    assert svg_output.startswith("<svg ")
    assert "Test sweep" in svg_output
    assert "tg c1 pp2048 tg128" in svg_output
    assert "tg c2 pp2048 tg128" in svg_output
    assert "generation t/s total" in svg_output


def test_sweep_outputs_preserve_prefill_when_generation_is_unavailable():
    results = BenchmarkResults()
    results.model_name = "diffusion-model"
    results.runs.append(
        BenchmarkRun(
            concurrency=1,
            context_size=2048,
            prompt_size=2048,
            response_size=128,
            is_context_prefill_phase=False,
            pp_throughput=metric(900.0),
            pp_req_throughput=metric(900.0),
            tg_throughput=None,
            tg_req_throughput=None,
            peak_throughput=metric(128.0),
            peak_req_throughput=metric(128.0),
            ttfr=metric(100.0),
            est_ppt=metric(90.0),
            e2e_ttft=metric(120.0),
        )
    )

    rows = list(csv.DictReader(io.StringIO(results._generate_sweep_csv())))
    assert len(rows) == 1
    assert rows[0]["ctx_tokens"] == "2048"
    assert rows[0]["prefill_tps"] == "900"
    assert rows[0]["gen_tps"] == ""

    svg_output = results._generate_sweep_svg("Prefill only")
    assert "Prefill only" in svg_output
    assert "pp c1 pp2048 tg128" in svg_output
    assert "generation t/s total" not in svg_output
