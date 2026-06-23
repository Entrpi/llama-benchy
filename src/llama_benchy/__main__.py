"""
Main entry point for the llama-benchy CLI.
"""

import asyncio
import datetime
import sys
from . import __version__
from .config import BenchmarkConfig
from .corpus import TokenizedCorpus
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner

async def main_async():
    # 1. Parse Configuration
    config = BenchmarkConfig.from_args()
    
    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    if config.prompt_suite:
        print(f"Prompt suite: {config.prompt_suite}")
    else:
        print(f"Concurrency levels: {config.concurrency_levels}")
    
    # 4. Initialize Components
    prompt_gen = None
    if not config.prompt_suite:
        # 3. Prepare Data
        corpus = TokenizedCorpus(config.book_url, config.tokenizer, config.model)
        print(f"Total tokens available in text corpus: {len(corpus)}")
        prompt_gen = PromptGenerator(corpus)

    client = LLMClient(config.base_url, config.api_key, config.served_model_name)
    runner = BenchmarkRunner(config, client, prompt_gen)
    
    # 5. Run Benchmark Suite
    await runner.run_suite()
    
    print(f"\nllama-benchy ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

def main():
    """Entry point for the CLI command."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
