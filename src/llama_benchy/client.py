import time
import json
import codecs
import aiohttp
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

_warned_about_fallback = False

@dataclass
class RequestResult:
    start_ts: float = 0.0
    end_ts: float = 0.0
    first_token_ts: Optional[float] = None
    first_response_ts: Optional[float] = None
    prompt_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    token_timestamps: List[float] = field(default_factory=list)

@dataclass
class PromptSuiteRequestResult:
    name: str
    run: int
    start_ts: float = 0.0
    end_ts: float = 0.0
    wall_s: float = 0.0
    prompt_tokens: int = 0
    predicted_n: int = 0
    predicted_per_second: float = 0.0
    draft_n: int = 0
    draft_n_accepted: int = 0
    accept_rate: Optional[float] = None
    error: Optional[str] = None

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def measure_latency(self, session: aiohttp.ClientSession, mode: str = "api") -> float:
        if mode == "none":
            print("Skipping latency measurement (assuming 0 ms).")
            return 0

        print(f"Measuring latency using mode: {mode}...")
        latencies = []
        
        for _ in range(3):
            start = time.perf_counter()
            try:
                if mode == "api":
                    async with session.get(f"{self.base_url}/models", headers=self.headers) as response:
                        await response.read()
                    latencies.append(time.perf_counter() - start)
                elif mode == "generation":
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "stream": True
                    }
                    async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                        async for _ in response.content:
                            latencies.append(time.perf_counter() - start)
                            break
                        async for _ in response.content: pass
            except Exception as e:
                print(f"Error measuring latency: {e}")
        
        if latencies:
            avg_latency = float(np.mean(latencies))
            print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
            return avg_latency
        return 0

    async def run_coherence_test(self, session: aiohttp.ClientSession) -> bool:
        """Run coherence test after warmup to verify model responds correctly."""
        print("\nRunning coherence test...")
        prompt = "What is the capital of France? Please reply with one word only"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }

        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                response_json = await response.json()

                if 'choices' not in response_json or len(response_json['choices']) == 0:
                    print("Coherence test FAILED: No choices in response")
                    return False

                choice = response_json['choices'][0].get('message', {})
                content = choice.get('content') or ''
                # Also check reasoning fields for thinking models
                reasoning = choice.get('reasoning') or choice.get('reasoning_content') or ''
                full_content = (content + reasoning).lower()

                if 'paris' in full_content:
                    print("Coherence test PASSED.")
                    return True
                else:
                    print(f"Coherence test FAILED: Expected 'Paris'. Got: {content[:200]}...")
                    return False
        except Exception as e:
            print(f"Coherence test FAILED with error: {e}")
            return False

    async def warmup(self, session: aiohttp.ClientSession, tokenizer=None):
        print("Warming up...")
        warmup_text = "Warmup " * 10

        delta_user = 0
        delta_context = 0

        # 1. User only
        payload_user = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": warmup_text}],
            "max_tokens": 1
        }

        try:
            async with session.post(f"{self.base_url}/chat/completions", json=payload_user, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Warmup failed: HTTP {response.status}: {error_text}")
                    raise SystemExit(1)
                response_json = await response.json()
                if tokenizer:
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        delta_user = prompt_tokens - local_tokens
                        print(f"Warmup (User only) complete. Delta: {delta_user} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                    else:
                        print("Warmup (User only) complete (no usage stats found).")
                else:
                    print("Warmup complete.")

            if tokenizer:
                # 2. Context Only
                payload_sys_empty = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": warmup_text},
                        {"role": "user", "content": ""}
                    ],
                    "max_tokens": 1
                }
                async with session.post(f"{self.base_url}/chat/completions", json=payload_sys_empty, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Warmup failed: HTTP {response.status}: {error_text}")
                        raise SystemExit(1)
                    response_json = await response.json()
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        delta_context = prompt_tokens - local_tokens
                        print(f"Warmup (System+Empty) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                    else:
                        delta_context = delta_user
        except Exception as e:
            print(f"Warmup failed: {e}")
            raise SystemExit(1)
        return delta_user, delta_context

    async def run_generation(
            self, 
            session: aiohttp.ClientSession, 
            context_text: str, 
            prompt_text: str, 
            max_tokens: int, 
            no_cache: bool,
            tokenizer=None
        ) -> RequestResult:

        messages = []
        if context_text:
            messages.append({"role": "system", "content": context_text})
        messages.append({"role": "user", "content": prompt_text})
        
        result = RequestResult()
        
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                "return_token_ids": True,
                "stream_options": {"include_usage": True},
            }
            
            if no_cache:
                payload["cache_prompt"] = False
            
            result.start_ts = time.perf_counter()

            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    result.end_ts = time.perf_counter()
                    result.error = f"HTTP {response.status}: {error_text}"
                    print(result.error)
                    return result

                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                buffer = ""
                
                async for chunk_bytes in response.content.iter_any():
                    chunk_time = time.perf_counter()
                    decoded_chunk = decoder.decode(chunk_bytes, final=False)
                    buffer += decoded_chunk
                    
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line == 'data: [DONE]' or line == 'data:[DONE]':
                            continue
                        
                        if line.startswith('data:'):
                            try:
                                json_str = line[5:].strip()
                                chunk = json.loads(json_str)

                                if 'usage' in chunk and chunk['usage'] is not None:
                                    result.prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
                                
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if result.first_response_ts is None:
                                        result.first_response_ts = chunk_time

                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    reasoning_content = delta.get('reasoning_content')
                                    reasoning = delta.get('reasoning')
                                    
                                    if content or reasoning_content or reasoning:
                                        if result.first_token_ts is None:
                                            result.first_token_ts = chunk_time
                                        
                                        token_ids = chunk['choices'][0].get('token_ids')
                                        if token_ids and isinstance(token_ids, list):
                                            result.total_tokens += len(token_ids)
                                            if len(token_ids) == 1:
                                                result.token_timestamps.append(chunk_time)
                                            else:
                                                last_ts = result.token_timestamps[-1] if result.token_timestamps else result.first_token_ts
                                                if last_ts is None:
                                                    last_ts = result.start_ts
                                                time_window = chunk_time - last_ts
                                                for i in range(len(token_ids)):
                                                    ts = last_ts + (time_window * (i + 1) / len(token_ids))
                                                    result.token_timestamps.append(ts)
                                        elif tokenizer is not None:
                                            global _warned_about_fallback
                                            if not _warned_about_fallback:
                                                print("  No token_ids in response, using local tokenization")
                                                _warned_about_fallback = True
                                            
                                            full_content = content or reasoning_content or reasoning
                                            token_count = len(tokenizer.encode(full_content, add_special_tokens=False))
                                            result.total_tokens += token_count
                                            if token_count == 1:
                                                result.token_timestamps.append(chunk_time)
                                            else:
                                                last_ts = result.token_timestamps[-1] if result.token_timestamps else result.first_token_ts
                                                if last_ts is None:
                                                    last_ts = result.start_ts
                                                time_window = chunk_time - last_ts
                                                for i in range(token_count):
                                                    ts = last_ts + (time_window * (i + 1) / token_count)
                                                    result.token_timestamps.append(ts)
                                        else:
                                            if not _warned_about_fallback:
                                                print("  No token_ids or tokenizer, assuming 1 token per chunk")
                                                _warned_about_fallback = True
                                            
                                            result.total_tokens += 1
                                            result.token_timestamps.append(chunk_time)
                            except json.JSONDecodeError:
                                continue
            
            result.end_ts = time.perf_counter()

        except Exception as e:
            print(f"Error during run: {e}")
            result.error = str(e)
        
        return result

    async def run_prompt_suite_generation(
            self,
            session: aiohttp.ClientSession,
            name: str,
            prompt: str,
            max_tokens: int,
            seed: Optional[int] = None,
            run: int = 1,
        ) -> PromptSuiteRequestResult:
        result = PromptSuiteRequestResult(name=name, run=run)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        try:
            result.start_ts = time.perf_counter()
            async with session.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    print(result.error)
                    return result

                response_json = await response.json()

            result.end_ts = time.perf_counter()
            result.wall_s = result.end_ts - result.start_ts

            usage = response_json.get("usage", {}) or {}
            timings = response_json.get("timings", {}) or {}

            result.prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
            result.predicted_n = int(
                usage.get("completion_tokens")
                or timings.get("predicted_n")
                or 0
            )
            result.predicted_per_second = float(
                timings.get("predicted_per_second")
                or (result.predicted_n / result.wall_s if result.wall_s > 0 else 0.0)
            )
            result.draft_n = int(timings.get("draft_n") or 0)
            result.draft_n_accepted = int(timings.get("draft_n_accepted") or 0)
            if result.draft_n:
                result.accept_rate = result.draft_n_accepted / result.draft_n

        except Exception as e:
            print(f"Error during prompt suite run: {e}")
            result.error = str(e)
            if result.start_ts:
                result.end_ts = time.perf_counter()
                result.wall_s = result.end_ts - result.start_ts

        return result
