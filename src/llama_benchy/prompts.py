import uuid
import numpy as np
from typing import Tuple, List, Dict

from .corpus import TokenizedCorpus


MTP_BENCH_PROMPTS: List[Dict[str, str]] = [
    {
        "name": "code_python",
        "prompt": "Write a Python function that returns the n-th Fibonacci number using memoization. Include a docstring.",
    },
    {
        "name": "code_cpp",
        "prompt": "Write a C++ template function `clamp(x, lo, hi)` that returns x clamped to [lo, hi]. No std::clamp.",
    },
    {
        "name": "explain_concept",
        "prompt": "Explain how speculative decoding works in large language model inference, in three short paragraphs.",
    },
    {
        "name": "summarize",
        "prompt": (
            "Summarize in two sentences: The Industrial Revolution began in Britain in the late 18th century, "
            "transforming manufacturing through mechanization, steam power, and the factory system. It spread "
            "to continental Europe and North America during the 19th century."
        ),
    },
    {
        "name": "qa_factual",
        "prompt": "Q: What are the four fundamental forces of physics?\nA:",
    },
    {
        "name": "translation",
        "prompt": "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    },
    {
        "name": "creative_short",
        "prompt": "Write a four-line poem about an old lighthouse.",
    },
    {
        "name": "stepwise_math",
        "prompt": (
            "Solve step by step: A train leaves station A at 60 km/h. Two hours later, a second train leaves "
            "the same station on the same track at 90 km/h. How long until the second train catches the first?"
        ),
    },
    {
        "name": "long_code_review",
        "prompt": (
            "You are reviewing a backend service that has been suffering intermittent latency spikes "
            "in production. Below is the relevant code and a description of the system. After reading "
            "carefully, produce a structured review with three sections: (1) likely root causes ranked "
            "by probability, (2) concrete code or configuration changes you would make first, "
            "(3) what telemetry you would add to confirm the diagnosis.\n\n"
            "System description: a Python FastAPI service in front of a Postgres 15 database, deployed "
            "as four replicas behind an nginx load balancer. Each request reads a user record, fetches "
            "their last 50 events from a partitioned events table, computes an aggregate score, writes "
            "the score back to the user row, and returns a JSON response. Average payload is 4 KB. "
            "p50 latency is 35 ms; p99 spikes to 1.8 seconds approximately every 90 seconds in a "
            "regular pattern. The spikes correlate with elevated Postgres CPU but not with elevated "
            "Postgres connection count. The application pool is sized at 20 connections per replica. "
            "PgBouncer is in front of Postgres in transaction pooling mode with a pool size of 50.\n\n"
            "Code excerpt - the hot endpoint:\n"
            "```python\n@app.post('/score/{user_id}')\nasync def score(user_id: int, payload: ScoreRequest):\n"
            "    async with db.transaction() as tx:\n        user = await tx.fetchrow(\n"
            "            'SELECT id, tier, last_score FROM users WHERE id = $1 FOR UPDATE',\n            user_id,\n        )\n"
            "        if user is None:\n            raise HTTPException(404)\n        events = await tx.fetch(\n"
            "            'SELECT type, weight, ts FROM events '\n            'WHERE user_id = $1 ORDER BY ts DESC LIMIT 50',\n            user_id,\n        )\n"
            "        new_score = compute_score(user['tier'], events, payload.signals)\n"
            "        await tx.execute(\n            'UPDATE users SET last_score = $1, updated_at = now() WHERE id = $2',\n            new_score, user_id,\n        )\n"
            "        await tx.execute(\n            'INSERT INTO score_history (user_id, score, ts) VALUES ($1, $2, now())',\n            user_id, new_score,\n        )\n"
            "    await cache.set(f'score:{user_id}', new_score, ex=300)\n"
            "    metrics.histogram('score.latency_ms').observe((time.time() - start) * 1000)\n"
            "    return {'user_id': user_id, 'score': new_score}\n```\n\n"
            "Schema notes: `users` is ~50M rows, `events` is partitioned by month with ~2B rows total "
            "and a btree index on `(user_id, ts DESC)`. `score_history` is unpartitioned, ~800M rows, "
            "with a single index on `user_id`. Postgres autovacuum is at default settings. There is "
            "a nightly batch job that rebuilds materialized views starting at 02:00 UTC; spikes occur "
            "throughout the day, not just during the batch window. Connection pooling metrics show "
            "PgBouncer waiting connections occasionally hit 8-12 during spikes but never saturate. "
            "CPU on the FastAPI replicas stays below 30% even during spikes. Network round-trip time "
            "between the application and Postgres is consistently 0.4 ms.\n\nBegin your review now."
        ),
    },
]

PROMPT_SUITES: Dict[str, List[Dict[str, str]]] = {
    "mtp-bench": MTP_BENCH_PROMPTS,
}

class PromptGenerator:
    def __init__(self, corpus: TokenizedCorpus):
        self.corpus = corpus
        self.tokenizer = corpus.get_tokenizer()
        self.all_tokens = corpus.get_tokens()

    def generate(self, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False) -> Tuple[str, str]:
        """
        Generates a single (context, prompt) pair.
        """
        suffix = ""
        suffix_len = 0
        if no_cache:
            suffix = f" {uuid.uuid4()}"
            suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))
        
        # Adjust prompt tokens to fetch from text
        text_prompt_tokens = max(0, prompt_tokens - suffix_len)
        
        # Create a pool of tokens large enough
        total_needed = text_prompt_tokens + context_tokens
        
        # Create a local reference to tokens to potentially extend
        current_tokens = self.all_tokens
        
        if len(current_tokens) < total_needed:
            # Repeat tokens if not enough
            current_tokens = current_tokens * (total_needed // len(current_tokens) + 2)
        
        # Pick a random start position
        max_start = len(current_tokens) - total_needed
        start_idx = np.random.randint(0, max_start)
        
        selected_tokens = current_tokens[start_idx : start_idx + total_needed]
        
        context_text = self.tokenizer.decode(selected_tokens[:context_tokens]) if context_tokens > 0 else ""
        prompt_text = self.tokenizer.decode(selected_tokens[context_tokens:])
        
        if no_cache:
            prompt_text += suffix
            
        return context_text, prompt_text

    def generate_batch(self, batch_size: int, prompt_tokens: int, context_tokens: int = 0, no_cache: bool = False) -> List[Tuple[str, str]]:
        """
        Generates a batch of (context, prompt) pairs.
        """
        return [self.generate(prompt_tokens, context_tokens, no_cache) for _ in range(batch_size)]
