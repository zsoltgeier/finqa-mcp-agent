import asyncio
import json
import time
from decimal import Decimal, ROUND_DOWN, localcontext
from collections import Counter
from functools import lru_cache

import httpx
from jsonschema import validate
from tokenizers import Tokenizer

from finqa_mcp.common import canonical_hash
from finqa_mcp.schemas import Answer, source_map
from finqa_mcp.bridge import action_schema, connect, inline_schema
from finqa_mcp.prompts import build_prompt
from finqa_mcp.report import full_report
from finqa_mcp.numbers import FACTORS


class RunFailure(Exception):
    pass


@lru_cache(maxsize=2)
def tokenizer(path):
    return Tokenizer.from_file(str(path))


def token_bound(messages, config):
    # No native tools: signatures are explicit system content and are counted here.
    rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    rendered += "<|im_start|>assistant\n<think>\n</think>\n\n"
    return len(tokenizer(config["tokenizer_path"]).encode(rendered).ids) + 256


def answer_schema(record):
    schema = inline_schema(Answer.model_json_schema())
    schema["properties"]["value"] = {"type": "string", "pattern": r"^-?[0-9]{1,40}(\.[0-9]{1,24})?$"}
    schema["properties"]["explanation"]["maxLength"] = 400
    schema["properties"]["sources"]["items"] = {"type": "string", "enum": list(source_map(record))}
    return schema


def decimal_output(value):
    with localcontext() as ctx:
        ctx.prec = 160
        value = Decimal(value).quantize(Decimal("1e-24"), rounding=ROUND_DOWN)
    text = format(value, "f").rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def grounded_answer_schema(record, calculations):
    """Constrain final numeric/unit pairs to actual tool outputs, never to references."""
    branches = []
    for calculation in calculations:
        if "error" in calculation:
            continue
        value, unit = Decimal(calculation["value"]), calculation["unit"]
        if unit in FACTORS:
            with localcontext() as ctx:
                ctx.prec = 160
                options = [(value * FACTORS[unit] / factor, [target]) for target, factor in FACTORS.items()]
        elif unit == "ratio":
            options = [(value, ["ratio", "number"]), (value * 100, ["percent", "percentage_point"])]
            # A division of unlabelled literal operands may be a native monetary amount.
            if not calculation.get("sources"):
                options.append((value, list(FACTORS) + ["years"]))
        else:
            options = [(value, list(answer_schema(record)["properties"]["unit"]["enum"]))]
        for converted, units in options:
            schema = answer_schema(record)
            schema["properties"]["value"] = {"type": "string", "enum": [decimal_output(converted)]}
            schema["properties"]["unit"] = {"type": "string", "enum": units}
            if calculation.get("sources"):
                schema["properties"]["sources"] = {"type": "array", "enum": [calculation["sources"]]}
            branches.append(schema)
    return {"anyOf": branches}


async def generate(client, config, messages, schema, result):
    estimate = token_bound(messages, config)
    if estimate + config["num_predict"] > config["num_ctx"]:
        raise RunFailure("context_budget")
    result["model_attempts"] += 1
    payload = {"model": config["model"], "messages": messages, "stream": False,
               "think": config["think"], "format": schema, "keep_alive": "30m",
               "options": {k: config[k] for k in ("num_ctx", "num_predict", "temperature", "seed")}}
    response = await client.post(config["ollama_url"] + "/api/chat", json=payload)
    if response.is_error:
        raise RunFailure(f"http_error: {response.status_code}: {response.text[:400]}")
    data = response.json()
    metadata = {k: data.get(k, 0) for k in ("prompt_eval_count", "eval_count", "load_duration", "total_duration", "done_reason")}
    metadata.update({"format_schema_sha256": canonical_hash(schema), "prompt_token_bound": estimate})
    result["model_calls"].append(metadata)
    message = data["message"]
    message.pop("thinking", None)
    messages.append(message)
    if data.get("prompt_eval_count", 0) > estimate:
        raise RunFailure("token_estimator_underflow")
    if data.get("done_reason") == "length":
        raise RunFailure("output_limit")
    if data.get("prompt_eval_count", 0) + config["num_predict"] >= config["num_ctx"]:
        raise RunFailure("context_budget")
    parsed = json.loads(message["content"])
    validate(parsed, schema)
    return parsed


def final_answer(value, record):
    answer = Answer.model_validate(value)
    if any(s not in source_map(record) for s in answer.sources):
        raise RunFailure("invalid_source")
    return answer.model_dump(mode="json")


async def tool_loop(record, config, demos, result, client, start):
    failure = None
    async with connect(config["input_dir"], record.id) as bridge:
        try:
            tools = bridge.tools
            result["discovered_tools"] = bridge.tools
            prompt = build_prompt("agent", demos) + "\nAVAILABLE MCP TOOL DEFINITIONS:\n" + json.dumps(tools, ensure_ascii=False)
            result["messages"].insert(0, {"role": "system", "content": prompt})
            schema = answer_schema(record)
            repeats, table_read, search_calls = Counter(), False, 0
            evidence_read, calculation_done, calculations = False, False, []
            async with asyncio.timeout(max(.01, config["timeout_seconds"] - (time.perf_counter() - start))):
                while True:
                    remaining = config["max_tool_calls"] - len(result["tool_calls"])
                    if not remaining and not calculation_done:
                        raise RunFailure("tool_budget")
                    available = [t for t in tools if not (table_read and t["function"]["name"] == "read_table")
                                 and not (not evidence_read and t["function"]["name"] == "calculate")
                                 and not (search_calls >= 2 and t["function"]["name"] == "search_report")]
                    if remaining == 1 and not calculation_done:
                        available = [t for t in available if t["function"]["name"] == "calculate"]
                    final_schema = grounded_answer_schema(record, calculations) if calculation_done else schema
                    output_schema = action_schema(available if remaining else [], final_schema, allow_final=calculation_done)
                    action = await generate(client, config, result["messages"], output_schema, result)
                    result["actions"].append(action)
                    if action["action"] == "final":
                        result["answer"] = final_answer(action["answer"], record)
                        result["status"] = "ok"
                        return
                    name, arguments = action["name"], action["arguments"]
                    key = canonical_hash([name, arguments])
                    repeats[key] += 1
                    if repeats[key] > config["max_repeat_calls"]:
                        raise RunFailure("repeated_call")
                    before = time.perf_counter()
                    output = await bridge.call(name, arguments)
                    if name == "search_report":
                        search_calls += 1
                    if name == "read_table" and "error" not in output:
                        table_read = True
                    if name in {"read_table", "search_report"} and "error" not in output:
                        evidence_read = True
                    if name == "calculate" and "error" not in output:
                        calculation_done = True
                        calculations.append(output)
                    result["tool_calls"].append({"index": len(result["tool_calls"]) + 1, "name": name,
                        "arguments": arguments, "result": output, "dispatched_via_mcp": bridge.last_dispatched,
                        "elapsed_seconds": time.perf_counter() - before})
                    remaining -= 1
                    feedback = {"example_id": record.id, "tool_name": name, "tool_result": output,
                                "remaining_tool_calls": remaining, "successful_calculation_available": calculation_done}
                    if repeats[key] > 1:
                        feedback["instruction"] = "This identical call was already made. Use the result; a third repeat stops the run."
                    if not remaining:
                        feedback["instruction"] = "No tool calls remain. Return action=final using the obtained results."
                    # The JSON action protocol has no native Ollama tool_call envelope.
                    # Deliver the observable MCP result as an explicit machine message.
                    result["messages"].append({"role": "user", "content": json.dumps(feedback, ensure_ascii=False)})
        except Exception as exc:
            failure = exc
    if failure:
        raise failure


async def run(record, variant, config, demonstrations):
    if variant not in {"baseline", "agent"}:
        raise ValueError("Unknown variant")
    start = time.perf_counter()
    question = {"example_id": record.id, "question": record.question}
    if variant == "baseline":
        question["report"] = full_report(record)
    result = {"id": record.id, "variant": variant, "status": "pending", "answer": None,
              "messages": [{"role": "user", "content": json.dumps(question, ensure_ascii=False)}],
              "model_calls": [], "model_attempts": 0, "tool_calls": [], "actions": []}
    async with httpx.AsyncClient(timeout=config["timeout_seconds"], trust_env=False) as client:
        try:
            if variant == "baseline":
                result["messages"].insert(0, {"role": "system", "content": build_prompt(variant, demonstrations)})
                async with asyncio.timeout(config["timeout_seconds"]):
                    value = await generate(client, config, result["messages"], answer_schema(record), result)
                    result["answer"] = final_answer(value, record)
                    result["status"] = "ok"
            else:
                await tool_loop(record, config, demonstrations, result, client, start)
        except (TimeoutError, httpx.TimeoutException):
            result["status"] = "timeout"
        except RunFailure as exc:
            result["status"] = str(exc).split(":")[0]
            result["error"] = str(exc)
        except Exception as exc:
            result["status"] = "invalid_format" if type(exc).__name__ in {"ValidationError", "JSONDecodeError"} else "technical_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = time.perf_counter() - start
    result["total_tokens"] = sum(c["prompt_eval_count"] + c["eval_count"] for c in result["model_calls"])
    result["load_seconds"] = sum(c["load_duration"] for c in result["model_calls"]) / 1e9
    return result
