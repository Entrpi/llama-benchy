import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json
import csv
import html
import io
import math
import sys

from .client import PromptSuiteRequestResult, RequestResult

# Type alias for a time series: List of [timestamp, value] pairs
TimeSeries = List[List[float]]

class BenchmarkMetric(BaseModel):
    mean: float = Field(..., description="Mean value")
    std: float = Field(..., description="Standard deviation")
    values: List[float] = Field(..., description="Raw values")

class BenchmarkMetadata(BaseModel):
    version: str = Field(..., description="Benchmark tool version")
    timestamp: str = Field(..., description="Run timestamp")
    latency_mode: str = Field(..., description="Latency measurement mode used")
    latency_ms: float = Field(..., description=" measured or assumed latency in ms")
    model: str = Field(..., description="Model name")
    prefix_caching_enabled: bool = Field(..., description="Whether prefix caching was enabled")
    max_concurrency: int = Field(..., description="Maximum concurrency level used in the suite")

class BenchmarkRun(BaseModel):
    concurrency: int = Field(..., description="Concurrency level for this run")
    context_size: int = Field(..., description="Context size (prefix tokens)")
    prompt_size: int = Field(..., description="Prompt size (tokens)")
    response_size: int = Field(..., description="Response size (tokens)")
    is_context_prefill_phase: bool = Field(..., description="Whether this was a context prefill phase run")
    
    # Metrics (using BenchmarkMetric)
    pp_throughput: Optional[BenchmarkMetric] = Field(None, description="Prefill tokens per second (total)")
    pp_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Prefill tokens per second (per request)")
    tg_throughput: Optional[BenchmarkMetric] = Field(None, description="Generation tokens per second (total)")
    tg_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Generation tokens per second (per request)")
    peak_throughput: Optional[BenchmarkMetric] = Field(None, description="Peak generation tokens per second (total)")
    peak_req_throughput: Optional[BenchmarkMetric] = Field(None, description="Peak generation tokens per second (per request)")
    ttfr: Optional[BenchmarkMetric] = Field(None, description="Time to First Response (ms)")
    est_ppt: Optional[BenchmarkMetric] = Field(None, description="Estimated pure processing time (ms)")
    e2e_ttft: Optional[BenchmarkMetric] = Field(None, description="End-to-end Time to First Token (ms)")
    
    # List of time series, one per run (aggregated across all requests in that run)
    throughput_over_time: Optional[List[TimeSeries]] = Field(
        None, 
        description="A collection of time series data capturing the aggregated throughput over time. Each item in the list represents a sequence of [timestamp, value] pairs for a specific execution batch or iteration."
    )
    
    # List of lists of time series, one list per run, containing one time series per request
    requests_throughput_over_time: Optional[List[List[TimeSeries]]] = Field(
        None, 
        description="A collection of time series data for individual requests. Organized as a list of lists, where the outer list represents batches and the inner list contains the throughput time series for each request in that batch."
    )

class BenchmarkReport(BenchmarkMetadata):
    benchmarks: List[BenchmarkRun] = Field(..., description="List of benchmark run results")

class PromptSuiteResults:
    def __init__(
        self,
        suite_name: str,
        model_name: str,
        max_tokens: int,
        seed: Optional[int],
        version: str,
        timestamp: str,
    ):
        self.suite_name = suite_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.seed = seed
        self.version = version
        self.timestamp = timestamp
        self.results: List[PromptSuiteRequestResult] = []

    def add(self, result: PromptSuiteRequestResult):
        self.results.append(result)

    def aggregate(self) -> Dict[str, Any]:
        valid = [result for result in self.results if not result.error]
        total_predicted = sum(result.predicted_n for result in valid)
        total_draft = sum(result.draft_n for result in valid)
        total_draft_accepted = sum(result.draft_n_accepted for result in valid)
        wall_s_total = sum(result.wall_s for result in valid)
        return {
            "n_requests": len(valid),
            "n_errors": len(self.results) - len(valid),
            "total_predicted": total_predicted,
            "total_draft": total_draft,
            "total_draft_accepted": total_draft_accepted,
            "aggregate_accept_rate": (
                round(total_draft_accepted / total_draft, 4) if total_draft else None
            ),
            "wall_s_total": round(wall_s_total, 3),
            "aggregate_predicted_per_second": (
                round(total_predicted / wall_s_total, 3) if wall_s_total > 0 else 0.0
            ),
        }

    def rows(self) -> List[Dict[str, Any]]:
        rows = []
        for result in self.results:
            rows.append({
                "suite": self.suite_name,
                "model": self.model_name,
                "run": result.run,
                "name": result.name,
                "wall_s": round(result.wall_s, 3),
                "prompt_tokens": result.prompt_tokens,
                "predicted_n": result.predicted_n,
                "predicted_per_second": round(result.predicted_per_second, 3),
                "draft_n": result.draft_n,
                "draft_n_accepted": result.draft_n_accepted,
                "accept_rate": round(result.accept_rate, 4) if result.accept_rate is not None else None,
                "error": result.error,
            })
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "suite": self.suite_name,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "results": self.rows(),
            "aggregate": self.aggregate(),
        }

    def _generate_md_report(self) -> str:
        rows = self.rows()
        if not rows:
            return "No prompt-suite results collected."

        data = [
            [
                row["run"],
                row["name"],
                row["predicted_n"],
                f'{row["wall_s"]:.3f}',
                f'{row["predicted_per_second"]:.2f}',
                row["draft_n"],
                row["draft_n_accepted"],
                f'{row["accept_rate"]:.3f}' if row["accept_rate"] is not None else "",
                row["error"] or "",
            ]
            for row in rows
        ]
        aggregate = self.aggregate()
        table = tabulate(
            data,
            headers=[
                "run",
                "prompt",
                "pred",
                "wall_s",
                "tok/s",
                "draft",
                "accepted",
                "accept",
                "error",
            ],
            tablefmt="pipe",
        )
        summary = (
            f"\nAggregate: {aggregate['aggregate_predicted_per_second']:.2f} tok/s, "
            f"wall={aggregate['wall_s_total']:.3f}s, "
            f"accept={aggregate['aggregate_accept_rate'] if aggregate['aggregate_accept_rate'] is not None else 'n/a'}"
        )
        return table + summary

    def _generate_csv(self) -> str:
        rows = self.rows()
        if not rows:
            return ""
        headers = [
            "suite",
            "model",
            "run",
            "name",
            "wall_s",
            "prompt_tokens",
            "predicted_n",
            "predicted_per_second",
            "draft_n",
            "draft_n_accepted",
            "accept_rate",
            "error",
        ]
        fp = io.StringIO()
        writer = csv.DictWriter(fp, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return fp.getvalue()

    def save_report(self, filename: Optional[str], format: str):
        print(
            f"{'Saving' if filename else 'Printing'} prompt-suite results"
            f"{f' to {filename}' if filename else ''} in {format.upper()} format...\n"
        )

        if format == "json":
            output = json.dumps(self.to_dict(), indent=2)
        elif format == "csv":
            output = self._generate_csv()
        elif format == "md":
            output = self._generate_md_report()
        else:
            raise ValueError(f"Unsupported prompt-suite output format: {format}")

        if filename:
            with open(filename, "w", encoding="utf-8", newline="" if format == "csv" else None) as f:
                f.write(output)
        else:
            print(output)

class BenchmarkResults:
    def __init__(self):
        self.runs: List[BenchmarkRun] = []
        self.metadata: Optional[BenchmarkMetadata] = None
        self.model_name: Optional[str] = None

    def _calculate_metric(self, values: List[float], multiplier: float = 1.0) -> Optional[BenchmarkMetric]:
        if not values:
            return None
        scaled_values = [v * multiplier for v in values]
        return BenchmarkMetric(
            mean=np.mean(values) * multiplier,
            std=np.std(values) * multiplier,
            values=scaled_values
        )

    def _calculate_peak_throughput(self, all_timestamps: List[float], window: float = 1.0, return_series: bool = False) -> Any:
        if not all_timestamps:
            return (0.0, []) if return_series else 0.0
        
        all_timestamps.sort()
        
        # If total duration is less than the window, use actual duration to calculate rate
        # This handles short bursts correctly where Peak would otherwise be < Mean
        total_duration = all_timestamps[-1] - all_timestamps[0]
        peak = 0.0
        if total_duration < window and total_duration > 0:
             peak = len(all_timestamps) / total_duration
             if not return_series:
                 return peak
             # If returning series, we should probably generate a flat series or just one point?
             # For now, let's proceed with sliding window calculation to fill the series, 
             # but use the adjusted peak if it's higher (it probably is). 
             # Actually, if duration < window, the loop below will just find max_tokens = len(all_timestamps)
             # and return len/window, which is smaller than len/duration.
             # So we must use the adjusted peak.

        max_tokens = 0
        series = []
        
        start_time = all_timestamps[0] if all_timestamps else 0

        start_idx = 0
        for end_idx, end_time in enumerate(all_timestamps):
            # Window starts at end_time - window
            while start_idx < end_idx and all_timestamps[start_idx] <= end_time - window:
                start_idx += 1
            
            # Count includes current token, so range is [start_idx, end_idx]
            current_tokens = end_idx - start_idx + 1
            if current_tokens > max_tokens:
                max_tokens = current_tokens
            
            if return_series:
                series.append([end_time - start_time, float(current_tokens) / window])
        
        calculated_peak = float(max_tokens) / window

        # If we had a short burst adjustment
        if total_duration < window and total_duration > 0:
             # Just use the adjusted peak as the result
             final_peak = peak
        else:
             final_peak = calculated_peak
             
        if return_series:
            return final_peak, series
        return final_peak

    def add(self, 
            model: str, 
            pp: int, 
            tg: int, 
            depth: int, 
            concurrency: int, 
            run_results: List[List[RequestResult]], # List of batches (one batch per run)
            latency: float, 
            expected_pp_tokens: int,
            is_context_phase: bool = False,
            save_total_throughput_timeseries: bool = False,
            save_all_throughput_timeseries: bool = False):
        
        if self.model_name is None:
            self.model_name = model

        # Aggregators
        agg_pp_speeds: List[float] = []
        agg_tg_speeds: List[float] = []
        agg_ttft_values: List[float] = []
        agg_ttfr_values: List[float] = []
        agg_est_ppt_values: List[float] = []
        agg_e2e_ttft_values: List[float] = []
        
        agg_batch_pp_throughputs: List[float] = []
        agg_batch_tg_throughputs: List[float] = []
        agg_peak_throughputs: List[float] = []
        agg_peak_req_throughputs: List[float] = []
        
        agg_throughput_series: List[TimeSeries] = []
        agg_req_throughput_series: List[List[TimeSeries]] = []

        for batch in run_results:
            self._process_batch(
                batch, 
                expected_pp_tokens, 
                latency, 
                agg_pp_speeds, 
                agg_tg_speeds, 
                agg_ttft_values, 
                agg_ttfr_values, 
                agg_est_ppt_values, 
                agg_e2e_ttft_values, 
                agg_batch_pp_throughputs, 
                agg_batch_tg_throughputs,
                agg_peak_throughputs,
                agg_peak_req_throughputs,
                save_total_throughput_timeseries=save_total_throughput_timeseries,
                save_all_throughput_timeseries=save_all_throughput_timeseries,
                agg_throughput_series=agg_throughput_series,
                agg_req_throughput_series=agg_req_throughput_series
            )

        # Calculate metrics for BenchmarkRun
        run_metric_pp_throughput = self._calculate_metric(agg_batch_pp_throughputs if concurrency > 1 else agg_pp_speeds)
        run_metric_pp_req_throughput = run_metric_pp_throughput if concurrency == 1 else self._calculate_metric(agg_pp_speeds)
        
        run_metric_tg_throughput = self._calculate_metric(agg_batch_tg_throughputs if concurrency > 1 else agg_tg_speeds)
        run_metric_tg_req_throughput = run_metric_tg_throughput if concurrency == 1 else self._calculate_metric(agg_tg_speeds)

        run_metric_peak_throughput = self._calculate_metric(agg_peak_throughputs)
        run_metric_peak_req_throughput = self._calculate_metric(agg_peak_req_throughputs)

        run_metric_ttfr = self._calculate_metric(agg_ttfr_values, 1000)
        run_metric_est_ppt = self._calculate_metric(agg_est_ppt_values, 1000)
        run_metric_e2e_ttft = self._calculate_metric(agg_e2e_ttft_values, 1000)

        self.runs.append(BenchmarkRun(
            concurrency=concurrency,
            context_size=depth,
            prompt_size=pp, # Configured prompt size
            response_size=tg,
            is_context_prefill_phase=is_context_phase,
            pp_throughput=run_metric_pp_throughput,
            pp_req_throughput=run_metric_pp_req_throughput,
            tg_throughput=run_metric_tg_throughput,
            tg_req_throughput=run_metric_tg_req_throughput,
            peak_throughput=run_metric_peak_throughput,
            peak_req_throughput=run_metric_peak_req_throughput,
            ttfr=run_metric_ttfr,
            est_ppt=run_metric_est_ppt,
            e2e_ttft=run_metric_e2e_ttft,
            throughput_over_time=agg_throughput_series if save_total_throughput_timeseries else None,
            requests_throughput_over_time=agg_req_throughput_series if save_all_throughput_timeseries else None
        ))

    def _process_batch(self, 
                       results: List[RequestResult], 
                       expected_pp_tokens: int, 
                       latency: float,
                       agg_pp_speeds: List[float],
                       agg_tg_speeds: List[float],
                       agg_ttft_values: List[float],
                       agg_ttfr_values: List[float],
                       agg_est_ppt_values: List[float],
                       agg_e2e_ttft_values: List[float],
                       agg_batch_pp_throughputs: List[float],
                       agg_batch_tg_throughputs: List[float],
                       agg_peak_throughputs: List[float],
                       agg_peak_req_throughputs: List[float],
                       save_total_throughput_timeseries: bool = False,
                       save_all_throughput_timeseries: bool = False,
                       agg_throughput_series: Optional[List[TimeSeries]] = None,
                       agg_req_throughput_series: Optional[List[List[TimeSeries]]] = None):
        
        valid_results = [r for r in results if r and not r.error]
        if not valid_results:
            return

        batch_prompt_tokens = 0
        batch_gen_tokens = 0
        
        start_times = []
        end_times = []
        first_token_times = []
        last_token_times = []
        
        # Collect all token timestamps for peak calculation
        all_token_timestamps = []
        
        batch_req_series = []

        for res in valid_results:
            start_times.append(res.start_ts)
            end_times.append(res.end_ts)
            all_token_timestamps.extend(res.token_timestamps)
            
            if save_all_throughput_timeseries:
                if res.token_timestamps:
                    # Calculate per-request throughput series
                    peak, series = self._calculate_peak_throughput(res.token_timestamps, return_series=True)
                    batch_req_series.append(series)
                    agg_peak_req_throughputs.append(peak)
                else:
                    batch_req_series.append([])
            elif res.token_timestamps:
                 peak = self._calculate_peak_throughput(res.token_timestamps, return_series=False)
                 agg_peak_req_throughputs.append(peak)

            if res.token_timestamps:
                last_token_times.append(res.token_timestamps[-1])
            elif res.end_ts:
                # Fallback if no timestamps recorded but request finished
                last_token_times.append(res.end_ts)
            
            # Use reported usage if available and reasonable, else expected
            prompt_tokens = expected_pp_tokens
            if res.prompt_tokens > 0:
                diff = abs(res.prompt_tokens - expected_pp_tokens)
                if diff < expected_pp_tokens * 0.2:
                    prompt_tokens = res.prompt_tokens
            
            batch_prompt_tokens += prompt_tokens
            batch_gen_tokens += res.total_tokens

            # Metrics Calculation
            ttft = 0.0
            e2e_ttft = 0.0
            ttfr = 0.0
            est_ppt = 0.0
            
            if res.first_response_ts:
                ttfr = res.first_response_ts - res.start_ts
                agg_ttfr_values.append(ttfr)
            
            if res.first_token_ts:
                first_token_times.append(res.first_token_ts)
                e2e_ttft = res.first_token_ts - res.start_ts
                ttft = max(0, e2e_ttft - latency)
                est_ppt = max(0, ttfr - latency)

                agg_e2e_ttft_values.append(e2e_ttft)
                agg_ttft_values.append(ttft)
                agg_est_ppt_values.append(est_ppt)

            # Individual Speeds
            if est_ppt > 0:
                pp_speed = prompt_tokens / est_ppt
                agg_pp_speeds.append(pp_speed)
            
            if res.total_tokens > 1 and res.first_token_ts:
                decode_time = res.end_ts - res.first_token_ts
                if decode_time > 0:
                    tg_speed = (res.total_tokens - 1) / decode_time
                    agg_tg_speeds.append(tg_speed)
        
        if save_all_throughput_timeseries and agg_req_throughput_series is not None:
             agg_req_throughput_series.append(batch_req_series)

        # Batch-Level Throughput
        if start_times and end_times and first_token_times:
            min_start = min(start_times)
            max_end = max(end_times)
            
            max_first_token = max(first_token_times)
            pp_duration = max_first_token - min_start
            
            if pp_duration > 0:
                batch_pp_throughput = batch_prompt_tokens / pp_duration
                agg_batch_pp_throughputs.append(batch_pp_throughput)
            
            min_first_token = min(first_token_times)
            
            # Use max(last_token_times) instead of max(end_times) to remove protocol overhead (headers, [DONE], etc)
            # This makes the throughput metric purely about token generation speed.
            max_last_token = max(last_token_times) if last_token_times else max_end
            tg_duration = max_last_token - min_first_token
            
            if tg_duration > 0:
                if batch_gen_tokens > len(valid_results):
                     batch_tg_throughput = (batch_gen_tokens - len(valid_results)) / tg_duration
                     agg_batch_tg_throughputs.append(batch_tg_throughput)

        if all_token_timestamps:
            res = self._calculate_peak_throughput(all_token_timestamps, return_series=save_total_throughput_timeseries)
            if save_total_throughput_timeseries:
                peak, series = res
                agg_peak_throughputs.append(peak)
                if agg_throughput_series is not None:
                    agg_throughput_series.append(series)
            else:
                agg_peak_throughputs.append(res)

    def _generate_sweep_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for run in self.runs:
            if run.is_context_prefill_phase:
                continue
            if not run.pp_throughput or not run.tg_throughput:
                continue

            rows.append({
                "ctx_tokens": run.context_size,
                "prefill_tokens": run.prompt_size,
                "prefill_tps": run.pp_throughput.mean,
                "gen_tokens": run.response_size,
                "gen_tps": run.tg_throughput.mean,
                "kvcache_bytes": "",
                "concurrency": run.concurrency,
                "prefill_tps_req": run.pp_req_throughput.mean if run.pp_req_throughput else "",
                "gen_tps_req": run.tg_req_throughput.mean if run.tg_req_throughput else "",
                "peak_gen_tps": run.peak_throughput.mean if run.peak_throughput else "",
                "peak_gen_tps_req": run.peak_req_throughput.mean if run.peak_req_throughput else "",
                "ttfr_ms": run.ttfr.mean if run.ttfr else "",
                "est_ppt_ms": run.est_ppt.mean if run.est_ppt else "",
                "e2e_ttft_ms": run.e2e_ttft.mean if run.e2e_ttft else "",
            })

        return sorted(
            rows,
            key=lambda row: (
                int(row["concurrency"]),
                int(row["prefill_tokens"]),
                int(row["gen_tokens"]),
                int(row["ctx_tokens"]),
            ),
        )

    def _generate_sweep_csv(self) -> str:
        rows = self._generate_sweep_rows()
        if not rows:
            return ""

        headers = [
            "ctx_tokens",
            "prefill_tokens",
            "prefill_tps",
            "gen_tokens",
            "gen_tps",
            "kvcache_bytes",
            "concurrency",
            "prefill_tps_req",
            "gen_tps_req",
            "peak_gen_tps",
            "peak_gen_tps_req",
            "ttfr_ms",
            "est_ppt_ms",
            "e2e_ttft_ms",
        ]

        def fmt_value(value: Any) -> Any:
            if isinstance(value, float):
                return f"{value:.6g}"
            return value

        fp = io.StringIO()
        writer = csv.DictWriter(fp, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt_value(row[key]) for key in headers})
        return fp.getvalue()

    def _generate_sweep_svg(self, title: Optional[str] = None, width: int = 960, height: int = 540) -> str:
        rows = self._generate_sweep_rows()
        if len(rows) < 1:
            return _render_empty_svg(title or "llama-benchy sweep", width, height)
        return _render_sweep_svg(rows, title or _derive_sweep_title(self.model_name), width, height)


    def _generate_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for run in self.runs:
            c_suffix = ""
            if self.metadata and self.metadata.max_concurrency > 1:
                c_suffix = f" (c{run.concurrency})"

            if run.is_context_prefill_phase:
                # Context Phase Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_pp @ d{run.context_size}{c_suffix}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "peak_ts": None,
                        "peak_ts_req": None,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Context Phase Token Generation
                if run.tg_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"ctx_tg @ d{run.context_size}{c_suffix}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "peak_ts": run.peak_throughput,
                        "peak_ts_req": run.peak_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
            else:
                # Standard Phase
                d_suffix = f" @ d{run.context_size}" if run.context_size > 0 else ""
                
                # Prompt Processing
                if run.pp_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"pp{run.prompt_size}{d_suffix}{c_suffix}",
                        "t_s": run.pp_throughput,
                        "t_s_req": run.pp_req_throughput,
                        "peak_ts": None,
                        "peak_ts_req": None,
                        "ttfr": run.ttfr,
                        "est_ppt": run.est_ppt,
                        "e2e_ttft": run.e2e_ttft
                    })
                
                # Token Generation
                if run.tg_throughput:
                    rows.append({
                        "model": self.model_name or "Unknown",
                        "test_name": f"tg{run.response_size}{d_suffix}{c_suffix}",
                        "t_s": run.tg_throughput,
                        "t_s_req": run.tg_req_throughput,
                        "peak_ts": run.peak_throughput,
                        "peak_ts_req": run.peak_req_throughput,
                        "ttfr": None,
                        "est_ppt": None,
                        "e2e_ttft": None
                    })
        return rows

    def _generate_md_report(self, concurrency: int) -> str:
        rows = self._generate_rows()
        if not rows:
            return "No results collected. Check if the model is generating tokens."

        def fmt(metric: Optional[BenchmarkMetric]) -> str:
            if metric is None:
                return ""
            return f"{metric.mean:.2f} ± {metric.std:.2f}"
            
        data = [[
            row["model"], 
            row["test_name"], 
            fmt(row["t_s"]), 
            fmt(row["t_s_req"]), 
            fmt(row["peak_ts"]),
            fmt(row["peak_ts_req"]),
            fmt(row["ttfr"]), 
            fmt(row["est_ppt"]), 
            fmt(row["e2e_ttft"])
        ] for row in rows]

        ts_header = "t/s (total)" if concurrency > 1 else "t/s"
        headers = ["model", "test", ts_header, "t/s (req)", "peak t/s", "peak t/s (req)", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]
        
        if concurrency == 1:
            data = [[
                row["model"], 
                row["test_name"], 
                fmt(row["t_s"]),
                fmt(row["peak_ts"]),
                fmt(row["ttfr"]), 
                fmt(row["est_ppt"]), 
                fmt(row["e2e_ttft"])
            ] for row in rows]
            headers = ["model", "test", ts_header, "peak t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"]

        return tabulate(data, headers=headers, tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right", "right", "right", "right") if concurrency > 1 else ("left", "right", "right", "right", "right", "right", "right"))

    def save_report(self, filename: Optional[str], format: str, concurrency: int = 1, sweep_title: Optional[str] = None):
        msg = ""
        if filename:
            msg += f"Saving results to {filename} in {format.upper()} format...\n"
        else:            
            msg += f"Printing results in {format.upper()} format:\n"

        print(f"{msg}\n")

        if format == "md":
            output = self._generate_md_report(concurrency)
            if filename:
                with open(filename, "w") as f:
                    f.write(output)
            else:
                 print("\n" + output)
        
        elif format == "json":
            output_data = {}
            # Flatten metadata if present
            if self.metadata:
                output_data.update(self.metadata.model_dump())
            
            # Serialize runs
            output_data["benchmarks"] = [run.model_dump() for run in self.runs]
            
            json_str = json.dumps(output_data, indent=2)
            
            if filename:
                 with open(filename, "w") as f:
                     f.write(json_str)
            else:
                 print(json_str)
        
        elif format == "csv":
             rows = self._generate_rows()
             csv_rows = []
             headers = ["model", "test_name", "t_s_mean", "t_s_std", "t_s_req_mean", "t_s_req_std", "peak_ts_mean", "peak_ts_std", "peak_ts_req_mean", "peak_ts_req_std", "ttfr_mean", "ttfr_std", "est_ppt_mean", "est_ppt_std", "e2e_ttft_mean", "e2e_ttft_std"]
             
             for r in rows:
                 row = {
                     "model": r["model"],
                     "test_name": r["test_name"],
                     "t_s_mean": r["t_s"].mean if r["t_s"] else None,
                     "t_s_std": r["t_s"].std if r["t_s"] else None,
                     "t_s_req_mean": r["t_s_req"].mean if r["t_s_req"] else None,
                     "t_s_req_std": r["t_s_req"].std if r["t_s_req"] else None,
                     "peak_ts_mean": r["peak_ts"].mean if r["peak_ts"] else None,
                     "peak_ts_std": r["peak_ts"].std if r["peak_ts"] else None,
                     "peak_ts_req_mean": r["peak_ts_req"].mean if r["peak_ts_req"] else None,
                     "peak_ts_req_std": r["peak_ts_req"].std if r["peak_ts_req"] else None,
                     "ttfr_mean": r["ttfr"].mean if r["ttfr"] else None,
                     "ttfr_std": r["ttfr"].std if r["ttfr"] else None,
                     "est_ppt_mean": r["est_ppt"].mean if r["est_ppt"] else None,
                     "est_ppt_std": r["est_ppt"].std if r["est_ppt"] else None,
                     "e2e_ttft_mean": r["e2e_ttft"].mean if r["e2e_ttft"] else None,
                     "e2e_ttft_std": r["e2e_ttft"].std if r["e2e_ttft"] else None,
                 }
                 csv_rows.append(row)
             
             if filename:
                 with open(filename, "w", newline="") as f:
                      writer = csv.DictWriter(f, fieldnames=headers)
                      writer.writeheader()
                      writer.writerows(csv_rows)
             else:
                 writer = csv.DictWriter(sys.stdout, fieldnames=headers)
                 writer.writeheader()
                 writer.writerows(csv_rows)

        elif format == "sweep-csv":
            output = self._generate_sweep_csv()
            if filename:
                with open(filename, "w", encoding="utf-8", newline="") as f:
                    f.write(output)
            else:
                print(output, end="")

        elif format in ("sweep-svg", "svg"):
            output = self._generate_sweep_svg(sweep_title)
            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(output)
            else:
                print(output, end="")


def _derive_sweep_title(model_name: Optional[str]) -> str:
    if not model_name:
        return "llama-benchy sweep"
    return f"{model_name} t/s"


def _nice_ceil(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = float(10 ** math.floor(math.log10(value)))
    normalized = value / magnitude
    for step in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0):
        if normalized <= step:
            return step * magnitude
    return 10.0 * magnitude


def _nice_step(span: float, target_ticks: int) -> float:
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = float(10 ** math.floor(math.log10(raw)))
    normalized = raw / magnitude
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= step:
            return step * magnitude
    return 10.0 * magnitude


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + step * 0.001:
        yield round(value, 10)
        value += step


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _project(point: Tuple[int, float], x_min: float, x_max: float, y_max: float, plot: Tuple[int, int, int, int]) -> str:
    left, top, width, height = plot
    x, y = point
    x_span = max(1.0, x_max - x_min)
    px = left + (x - x_min) / x_span * width
    py = top + height - y / y_max * height
    return f"{px:.2f},{py:.2f}"


def _polyline(points: List[Tuple[int, float]], x_min: float, x_max: float, y_max: float, plot: Tuple[int, int, int, int]) -> str:
    return " ".join(_project(point, x_min, x_max, y_max, plot) for point in points)


def _render_empty_svg(title: str, width: int, height: int) -> str:
    safe_title = html.escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#1f2933">{safe_title}: no sweep data</text>\n'
        "</svg>\n"
    )


def _render_sweep_svg(rows: List[Dict[str, Any]], title: str, width: int, height: int) -> str:
    margin_left = 82
    margin_right = 92
    margin_top = 66
    margin_bottom = 72
    plot = (
        margin_left,
        margin_top,
        width - margin_left - margin_right,
        height - margin_top - margin_bottom,
    )
    left, top, plot_width, plot_height = plot
    right = left + plot_width
    bottom = top + plot_height

    groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["concurrency"]), int(row["prefill_tokens"]), int(row["gen_tokens"]))
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row["ctx_tokens"]))

    ctx_values = [int(row["ctx_tokens"]) for row in rows]
    prefill_values = [float(row["prefill_tps"]) for row in rows]
    gen_values = [float(row["gen_tps"]) for row in rows]

    x_min = 0
    x_max = max(ctx_values)
    prefill_max = _nice_ceil(max(prefill_values) * 1.05)
    gen_max = _nice_ceil(max(gen_values) * 1.05)

    x_step = _nice_step(x_max - x_min, 6)
    x_ticks = []
    tick = math.ceil(x_min / x_step) * x_step
    while tick <= x_max:
        x_ticks.append(tick)
        tick += x_step

    prefill_step = _nice_step(prefill_max, 5)
    gen_step = _nice_step(gen_max, 5)
    prefill_ticks = list(_frange(0, prefill_max, prefill_step))
    gen_ticks = list(_frange(0, gen_max, gen_step))

    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be123c", "#4b5563"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
    ]

    for tick in prefill_ticks:
        y = bottom - tick / prefill_max * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{_fmt_tick(tick)}</text>')

    for tick in gen_ticks:
        y = bottom - tick / gen_max * plot_height
        parts.append(f'<text x="{right + 10}" y="{y + 4:.2f}" text-anchor="start" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{_fmt_tick(tick)}</text>')

    for tick in x_ticks:
        x = left + (tick - x_min) / max(1.0, x_max - x_min) * plot_width
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{_fmt_tick(tick)}</text>')

    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155" stroke-width="1.5"/>',
        f'<line x1="{right}" y1="{top}" x2="{right}" y2="{bottom}" stroke="#334155" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155" stroke-width="1.5"/>',
        f'<text x="{(left + right) / 2:.0f}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#1f2933">context depth tokens</text>',
        f'<text transform="translate(22 {(top + bottom) / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#2563eb">prefill t/s total</text>',
        f'<text transform="translate({width - 22} {(top + bottom) / 2:.0f}) rotate(90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#dc2626">generation t/s total</text>',
    ])

    legend_x = right - 210
    legend_y = top + 18
    line_idx = 0
    for idx, (key, group_rows) in enumerate(sorted(groups.items())):
        concurrency, prefill_tokens, gen_tokens = key
        color = palette[idx % len(palette)]
        label = f"c{concurrency} pp{prefill_tokens} tg{gen_tokens}"
        prefill_points = [(int(row["ctx_tokens"]), float(row["prefill_tps"])) for row in group_rows]
        gen_points = [(int(row["ctx_tokens"]), float(row["gen_tps"])) for row in group_rows]
        parts.append(f'<polyline points="{_polyline(prefill_points, x_min, x_max, prefill_max, plot)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<polyline points="{_polyline(gen_points, x_min, x_max, gen_max, plot)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="7 4"/>')
        y = legend_y + line_idx * 20
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<line x1="{legend_x + 78}" y1="{y}" x2="{legend_x + 100}" y2="{y}" stroke="{color}" stroke-width="2.5" stroke-dasharray="7 4"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#1f2933">pp</text>')
        parts.append(f'<text x="{legend_x + 106}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#1f2933">tg {html.escape(label)}</text>')
        line_idx += 1

    parts.append("</svg>")
    return "\n".join(parts) + "\n"
