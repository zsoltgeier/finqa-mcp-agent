import asyncio

import httpx
import pytest

from finqa_mcp import runner
from finqa_mcp.common import canonical_hash, write_json
from finqa_mcp.schemas import InputRecord


@pytest.mark.parametrize("variant", ["baseline", "agent"])
def test_complete_run_with_mock_model_and_real_mcp(variant, tmp_path, monkeypatch):
    record = InputRecord(id="synthetic", question="What was the percentage increase in revenue?",
                         pre_text=[], post_text=[], table=[["", "2020", "2019"], ["Revenue", "120", "100"]])
    write_json(tmp_path / f"{canonical_hash(record.id)}.json", record.model_dump())
    answer = {"value": "20", "unit": "percent", "explanation": "Revenue increased by 20%.",
              "sources": ["table_r1_c1", "table_r1_c2"]}
    actions = []
    if variant == "agent":
        actions.append({"action": "tool", "name": "read_table", "arguments": {"example_id": record.id}})
    if variant != "baseline":
        actions.extend([
            {"action": "tool", "name": "calculate", "arguments": {"operation": "subtract", "operands": ["table_r1_c1", "table_r1_c2"]}},
            {"action": "tool", "name": "calculate", "arguments": {"operation": "divide", "operands": ["calc_1", "table_r1_c2"]}},
            {"action": "final", "answer": answer}])
    else:
        actions.append(answer)
    import json
    responses = iter(actions)

    def model(request):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(next(responses))},
                                        "prompt_eval_count": 20, "eval_count": 10, "done_reason": "stop"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(runner.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(model), **kwargs))
    monkeypatch.setattr(runner, "token_bound", lambda messages, config: 100)
    config = {"input_dir": str(tmp_path), "ollama_url": "http://local.invalid", "model": "mock",
              "num_ctx": 16384, "num_predict": 512, "think": False, "temperature": 0, "seed": 0,
              "timeout_seconds": 30, "max_tool_calls": 6, "max_repeat_calls": 2}
    result = asyncio.run(runner.run(record, variant, config, {"examples": []}))
    assert result["status"] == "ok", result
    assert result["answer"] == answer
    assert len(result["model_calls"]) == len(actions)
    assert all(call["dispatched_via_mcp"] for call in result["tool_calls"])
    if variant != "baseline":
        assert result["tool_calls"][-1]["result"]["percent_value"] == "20.0"
