import json

from llama_benchy.progress import ProgressEmitter, SCHEMA_VERSION


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_progress_emitter_writes_estimated_tokens_and_terminal_status(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    emitter = ProgressEmitter(str(progress_path), llama_benchy_version="test-version")

    emitter.tokens(request_id=3, count=1, snippet="hello", estimated=True)
    emitter.tokens(request_id=3, count=2, snippet=" world")
    emitter.bench_complete(status="interrupted")
    emitter.close()

    events = _read_jsonl(progress_path)

    assert events[0]["schema"] == SCHEMA_VERSION
    assert events[0]["type"] == "header"
    assert events[0]["llama_benchy_version"] == "test-version"

    assert events[1]["type"] == "tokens"
    assert events[1]["estimated"] is True

    assert events[2]["type"] == "tokens"
    assert "estimated" not in events[2]

    assert events[3]["type"] == "bench_complete"
    assert events[3]["status"] == "interrupted"
