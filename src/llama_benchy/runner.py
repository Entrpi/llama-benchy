import asyncio
import subprocess
import time
import sys
from datetime import datetime, timezone
from typing import List, Optional
import aiohttp

from ._version import __version__
from .config import BenchmarkConfig
from .client import CONTEXT_LOAD_USER_MESSAGE, LLMClient, PromptSuiteRequestResult
from .prompts import PROMPT_SUITES, PromptGenerator
from .results import BenchmarkResults, BenchmarkMetadata, PromptSuiteResults

class BenchmarkFailure(Exception):
    pass

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, client: LLMClient, prompt_generator: Optional[PromptGenerator], progress=None):
        self.config = config
        self.client = client
        self.prompt_gen = prompt_generator
        self.results = BenchmarkResults()
        self.progress = progress
        self._next_request_id = 0

        # We need to track deltas from warmup to adapt prompts
        self.delta_user = 0
        self.delta_context = 0

    def _new_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid

    def _emit_request_start(
        self,
        request_id: int,
        pp: int,
        tg: int,
        depth: int,
        concurrency: int,
        run_index: int,
        target_label: str = "",
    ) -> None:
        if self.progress is None:
            return
        try:
            self.progress.request_start(
                request_id=request_id,
                model=self.config.model,
                base_url=self.config.base_url,
                prompt_size=pp,
                response_size=tg,
                context_size=depth,
                concurrency=concurrency,
                run_index=run_index,
                target_label=target_label,
            )
        except Exception:
            pass

    def _emit_prompt_suite_request_end(
        self,
        request_id: int,
        result: PromptSuiteRequestResult,
    ) -> None:
        if self.progress is None:
            return
        try:
            self.progress.request_end(
                request_id=request_id,
                total_tokens=result.predicted_n,
                prompt_tokens=result.prompt_tokens,
                decode_seconds=0.0,
                error=result.error or "",
            )
        except Exception:
            pass

    async def run_suite(self):
        if self.config.prompt_suite:
            await self.run_prompt_suite()
            return

        prompt_gen = self.prompt_gen
        if prompt_gen is None:
            raise RuntimeError("Prompt generator is required for synthetic benchmark runs")

        # Initialize session
        timeout = aiohttp.ClientTimeout(total=3600)
        max_concurrency = max(self.config.concurrency_levels)
        connector = aiohttp.TCPConnector(limit=max_concurrency + 5, force_close=False, keepalive_timeout=600)
        latency = 0.0  # default in case of early interrupt

        try:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
                # Warmup
                should_warmup = not self.config.no_warmup
                if self.config.adapt_prompt:
                    should_warmup = True

                tokenizer = prompt_gen.corpus.get_tokenizer()

                if should_warmup:
                    self.delta_user, self.delta_context = await self.client.warmup(session, tokenizer)

                # Coherence test after warmup (by default, unless skipped)
                if not self.config.skip_coherence:
                    if not await self.client.run_coherence_test(session):
                        print("\nBenchmark failed due to coherence test failure.")
                        raise SystemExit(1)
                else:
                    print("\nSkipping coherence test (--skip-coherence specified)")

                # Measure latency
                latency = await self.client.measure_latency(session, self.config.latency_mode)
                if self.progress is not None:
                    try:
                        self.progress.latency_measured(
                            latency_s=latency, mode=self.config.latency_mode
                        )
                    except Exception:
                        pass

                # Main Loop
                for depth in self.config.depths:
                    for pp in self.config.pp_counts:
                        for tg in self.config.tg_counts:
                            for concurrency in self.config.concurrency_levels:
                                print(f"Running test: pp={pp}, tg={tg}, depth={depth}, concurrency={concurrency}")

                                run_std_results = []
                                run_ctx_results = []
                                expected_pp = pp
                                expected_ctx = depth

                                total_runs = self.config.num_runs if self.config.no_warmup else self.config.num_runs + 1
                                for run in range(total_runs):
                                    is_warmup = not self.config.no_warmup and run == 0
                                    run_label = "Warmup" if is_warmup else f"Run {run if not self.config.no_warmup else run + 1}/{self.config.num_runs}"

                                    # Adapt prompt tokens
                                    current_pp = pp
                                    current_depth = depth
                                    if self.config.adapt_prompt:
                                        if depth == 0:
                                            current_pp = max(1, pp - self.delta_user)
                                        else:
                                            current_depth = max(1, depth - self.delta_context)

                                    expected_pp = current_pp
                                    expected_ctx = current_depth

                                    prompt_batch = prompt_gen.generate_batch(
                                        concurrency,
                                        current_pp,
                                        current_depth,
                                        self.config.no_cache
                                    )

                                    if self.config.enable_prefix_caching and depth > 0:
                                        # Phase 1: Context Load
                                        print(f"  {run_label} (Context Load, batch size {concurrency})...")
                                        load_tasks = []
                                        for i in range(concurrency):
                                            context, _ = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, run)
                                            load_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=CONTEXT_LOAD_USER_MESSAGE,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        load_results = await asyncio.gather(*load_tasks)
                                        if not is_warmup:
                                            run_ctx_results.append(load_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in load_results):
                                            first_error = next(r.error for r in load_results if r.error)
                                            print(f"\n[Error] Stopping due to error in context load: {first_error}")
                                            raise BenchmarkFailure()

                                        # Phase 2: Inference
                                        print(f"  {run_label} (Inference, batch size {concurrency})...")
                                        inf_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, run)
                                            inf_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        batch_results = await asyncio.gather(*inf_tasks)
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in inference: {first_error}")
                                            raise BenchmarkFailure()

                                    else:
                                        # Standard Run
                                        print(f"  {run_label} (batch size {concurrency})...")
                                        expected_tokens = current_pp + current_depth
                                        batch_tasks = []
                                        for i in range(concurrency):
                                            context, prompt = prompt_batch[i]
                                            if not is_warmup:
                                                rid = self._new_request_id()
                                                self._emit_request_start(rid, pp, tg, depth, concurrency, run)
                                            batch_tasks.append(self.client.run_generation(
                                                session,
                                                context_text=context,
                                                prompt_text=prompt,
                                                max_tokens=tg,
                                                no_cache=self.config.no_cache,
                                                tokenizer=tokenizer,
                                                progress=None if is_warmup else self.progress,
                                                request_id=None if is_warmup else rid,
                                            ))

                                        batch_results = await asyncio.gather(*batch_tasks)
                                        if not is_warmup:
                                            run_std_results.append(batch_results)

                                        if self.config.exit_on_first_fail and any(r.error for r in batch_results):
                                            first_error = next(r.error for r in batch_results if r.error)
                                            print(f"\n[Error] Stopping due to error in standard run: {first_error}")
                                            raise BenchmarkFailure()


                                    # Post Run Command
                                    if self.config.post_run_cmd:
                                        try:
                                            subprocess.run(self.config.post_run_cmd, shell=True, check=True)
                                        except subprocess.CalledProcessError as e:
                                            print(f"Post-run command failed: {e}")

                                # Aggregate and Record
                                if self.config.enable_prefix_caching and depth > 0:
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_ctx_results, latency, expected_ctx, is_context_phase=True, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)
                                else:
                                    # Standard run expected tokens = pp + depth (usually depth=0 or concatenated)
                                    # In the loop above: expected_tokens = current_pp + current_depth
                                    self.results.add(self.config.model, pp, tg, depth, concurrency, run_std_results, latency, expected_pp + expected_ctx, is_context_phase=False, save_total_throughput_timeseries=self.config.save_total_throughput_timeseries, save_all_throughput_timeseries=self.config.save_all_throughput_timeseries)

                self.results.metadata = BenchmarkMetadata(
                    version=__version__,
                    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                    latency_mode=self.config.latency_mode,
                    latency_ms=latency * 1000,
                    model=self.config.model,
                    prefix_caching_enabled=self.config.enable_prefix_caching,
                    max_concurrency=max(self.config.concurrency_levels) if self.config.concurrency_levels else 1
                )

                self._save_results(max(self.config.concurrency_levels) if self.config.concurrency_levels else 1)

        except (asyncio.CancelledError, KeyboardInterrupt, BenchmarkFailure) as e:
            if self.results.runs:
                should_save = True
                if isinstance(e, BenchmarkFailure) and self.config.no_results_on_fail:
                    should_save = False
                    print("\n[Failed] Results discarded per --no-results-on-fail.")

                if should_save:
                    print("\n[Interrupted/Failed] Saving partial results...")
                    if self.results.metadata is None:
                        self.results.metadata = BenchmarkMetadata(
                            version=__version__,
                            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                            latency_mode=self.config.latency_mode,
                            latency_ms=latency * 1000,
                            model=self.config.model,
                            prefix_caching_enabled=self.config.enable_prefix_caching,
                            max_concurrency=max_concurrency
                        )
                    self._save_results(max_concurrency)

            if isinstance(e, BenchmarkFailure):
                sys.exit(1)
            raise

    async def run_prompt_suite(self):
        prompts = PROMPT_SUITES.get(self.config.prompt_suite or "")
        if prompts is None:
            raise ValueError(f"Unknown prompt suite: {self.config.prompt_suite}")

        timeout = aiohttp.ClientTimeout(total=3600)
        connector = aiohttp.TCPConnector(limit=2, force_close=False, keepalive_timeout=600)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        suite_results = PromptSuiteResults(
            suite_name=self.config.prompt_suite or "",
            model_name=self.config.model,
            max_tokens=self.config.suite_max_tokens,
            seed=self.config.suite_seed,
            version=__version__,
            timestamp=timestamp,
        )

        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
            for run in range(1, self.config.suite_runs + 1):
                for prompt in prompts:
                    print(f"Running prompt suite: {self.config.prompt_suite} run={run} prompt={prompt['name']}")
                    request_id = self._new_request_id()
                    self._emit_request_start(
                        request_id,
                        pp=0,
                        tg=self.config.suite_max_tokens,
                        depth=0,
                        concurrency=1,
                        run_index=run - 1,
                        target_label=prompt["name"],
                    )
                    result = await self.client.run_prompt_suite_generation(
                        session,
                        name=prompt["name"],
                        prompt=prompt["prompt"],
                        max_tokens=self.config.suite_max_tokens,
                        seed=self.config.suite_seed,
                        run=run,
                    )
                    self._emit_prompt_suite_request_end(request_id, result)
                    suite_results.add(result)
                    if result.error:
                        if self.config.exit_on_first_fail:
                            break
                        continue

                    accept_rate = f"{result.accept_rate:.3f}" if result.accept_rate is not None else "n/a"
                    print(
                        f"  {result.name:<18} pred={result.predicted_n:>4} "
                        f"draft={result.draft_n:>4} acc={result.draft_n_accepted:>4} "
                        f"rate={accept_rate} tok/s={result.predicted_per_second:.1f}"
                    )
                if self.config.exit_on_first_fail and any(result.error for result in suite_results.results):
                    break

        suite_results.save_report(self.config.save_result, self.config.result_format)

        if self.config.exit_on_first_fail and any(result.error for result in suite_results.results):
            raise BenchmarkFailure()

    def _save_results(self, max_concurrency: int):
        self.results.save_report(
            self.config.save_result,
            self.config.result_format,
            max_concurrency,
            self.config.sweep_title,
        )

        if self.config.sweep_csv:
            self.results.save_report(
                self.config.sweep_csv,
                "sweep-csv",
                max_concurrency,
                self.config.sweep_title,
            )

        if self.config.sweep_svg:
            self.results.save_report(
                self.config.sweep_svg,
                "sweep-svg",
                max_concurrency,
                self.config.sweep_title,
            )
