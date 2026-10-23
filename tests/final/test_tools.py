import asyncio
import json
from decimal import Decimal

import pytest
from jsonschema import ValidationError, validate

from finqa_mcp.common import canonical_hash, write_json
from finqa_mcp.schemas import InputRecord
from finqa_mcp.bridge import action_schema, connect
from finqa_mcp.numbers import Calculator, NumberError, parse_number, text_numbers
from finqa_mcp.runner import answer_schema, decimal_output, grounded_answer_schema


@pytest.mark.parametrize("original,value,unit", [
    ("$1,234.50", "1234.50", "USD"), ("(2.3)", "-2.3", "number"),
    ("$ -156 ( 156 )", "-156", "USD"), ("36.3% ( 36.3 % )", ".363", "ratio"),
    ("( 60.8 ) % ( % )", "-.608", "ratio"), ("$ 1.2 billion", "1.2", "billion_USD"),
    ("2014", "2014", "number"), ("-0.1 ( 0.1 )", "-.1", "number")])
def test_financial_parser(original, value, unit):
    parsed = parse_number(original)
    assert parsed.value == Decimal(value) and parsed.unit == unit
    assert parsed.original == original


@pytest.mark.parametrize("value", ["-", "NaN", "Infinity", "1,2", "1/2", "__import__('os')", "1e101", "2 (3)"])
def test_parser_rejects_ambiguous_or_unsafe(value):
    with pytest.raises(NumberError):
        parse_number(value)


@pytest.fixture
def record():
    return InputRecord(id="ACME/2020/page_1.pdf-1", question="What was the percentage increase in revenue?",
        pre_text=["All figures in millions of dollars."],
        post_text=["Expenses were $ 3.2 million, $ 23.3 million and $ 4.4 million in 2020, 2019 and 2018."],
        table=[["", "2020", "2019"], ["Revenue", "$120", "$100"], ["Rate", "36.3% ( 36.3 % )", "32.9% ( 32.9 % )"]])


def test_cell_references_units_and_provenance(record):
    calc = Calculator(record)
    first = calc.calculate("subtract", ["table_r1_c1", {"source_id": "table_r1_c2"}])
    assert first["value"] == "20" and first["unit"] == "million_USD"
    second = calc.calculate("divide", [first["result_id"], "table_r1_c2"])
    assert Decimal(second["value"]) == Decimal(".2") and Decimal(second["percent_value"]) == 20
    assert second["sources"] == ["table_r1_c1", "table_r1_c2"]
    rate = calc.calculate("subtract", ["table_r2_c1", "table_r2_c2"])
    assert Decimal(rate["value"]) == Decimal(".034") and Decimal(rate["percent_value"]) == Decimal("3.4")


def test_text_reference_mean_and_errors(record):
    calc = Calculator(record)
    numbers = calc.numeric_values("post_0")
    assert [n["value"] for n in numbers[:3]] == ["3.2", "23.3", "4.4"]
    first = calc.calculate("add", [{"source_id": "post_0", "number_index": 0}, {"source_id": "post_0", "number_index": 1}])
    second = calc.calculate("add", [first["result_id"], {"source_id": "post_0", "number_index": 2}])
    third = calc.calculate("divide", [second["result_id"], "3"])
    assert third["value"] == "10.3" and third["unit"] == "million_USD" and third["sources"] == ["post_0"]
    assert calc.calculate("divide", ["1", "0"])["error"] == "division_by_zero"
    assert calc.calculate("add", ["calc_99", "1"])["error"] == "unknown_result"
    assert calc.calculate("add", [{"source_id": "../../references.json"}, "1"])["error"] == "unknown_source"
    assert calc.calculate("add", [{"source_id": "post_0", "number_index": 99}, "1"])["error"] == "invalid_number_index"


def test_text_duplicate_percent_is_one_occurrence():
    assert text_numbers("rate 36.3% (36.3 %) in 2020") == ["36.3%", "2020"]


def test_currency_scale_conversion(record):
    calc = Calculator(record)
    result = calc.calculate("add", ["$1 million", "$500 thousand"])
    assert Decimal(result["value"]) == Decimal("1.5") and result["unit"] == "million_USD"


def test_currency_implied_across_table_and_mixed_cell():
    record = InputRecord(id="currency", question="current ratio", pre_text=["in thousands"], post_text=[],
        table=[["", "value"], ["current assets", "$ 513782"], ["current liabilities", "310919"],
               ["face value", "20ac750 ( approximately $ 1029 )"]])
    calc = Calculator(record)
    result = calc.calculate("divide", ["table_r1_c1", "table_r2_c1"])
    assert result["unit"] == "ratio" and abs(Decimal(result["value"]) - Decimal("1.65246")) < Decimal(".00001")
    values = calc.numeric_values("table_r3_c1")
    assert len(values) == 2 and values[1]["value"] == "1029" and values[1]["unit"] == "thousand_USD"


def test_actual_mcp_and_tool_schema(record, tmp_path):
    write_json(tmp_path / f"{canonical_hash(record.id)}.json", record.model_dump())

    async def check():
        async with connect(tmp_path, record.id) as bridge:
            text = await bridge.call("search_report", {"example_id": record.id, "query": "Expenses", "top_k": 1})
            assert text["matches"][0]["source_id"] == "post_0"
            assert text["matches"][0]["numeric_values"][0]["value"] == "3.2"
            table = await bridge.call("read_table", {"example_id": record.id})
            assert table["table"][1]["cells"][1]["original"] == "$120"
            result = await bridge.call("calculate", {"operation": "divide", "operands": ["table_r1_c1", "table_r1_c2"]})
            assert Decimal(result["value"]) == Decimal("1.2") and bridge.last_dispatched
            assert (await bridge.call("read_table", {"example_id": "other"}))["error"] == "access_denied"
            assert (await bridge.call("search_report", {"example_id": "other", "query": "x"}))["error"] == "access_denied"
            schema = action_schema([t for t in bridge.tools if t["function"]["name"] == "calculate"], answer_schema(record))
            validate({"action": "tool", "name": "calculate", "arguments": {"operation": "add", "operands": ["1", {"source_id": "table_r1_c1"}]}}, schema)
            with pytest.raises(ValidationError):
                validate({"action": "tool", "name": "read_table", "arguments": {"example_id": record.id}}, schema)
            final = {"action": "final", "answer": {"value": "20", "unit": "percent", "explanation": "growth", "sources": ["table_r1_c1"]}}
            validate(final, schema)
            with pytest.raises(ValidationError):
                validate({**final, "answer": {**final["answer"], "sources": ["unknown"]}}, schema)
            with pytest.raises(ValidationError):
                validate({"action": "tool", "name": "calculate", "arguments": {"operation": "add", "operands": ["1", "2", "3"]}}, schema)
    asyncio.run(check())


def test_schema_forbids_ungrounded_final(record):
    tool = {"type": "function", "function": {"name": "read_table", "parameters": {
        "type": "object", "properties": {"example_id": {"type": "string"}}, "required": ["example_id"]}}}
    schema = action_schema([tool], answer_schema(record), allow_final=False)
    validate({"action": "tool", "name": "read_table", "arguments": {"example_id": record.id}}, schema)
    with pytest.raises(ValidationError):
        validate({"action": "final", "answer": {"value": "1", "unit": "number", "explanation": "guess", "sources": ["pre_0"]}}, schema)


def test_final_schema_bounds_recurring_decimal(record):
    good = {"value": "0.123456789012345678901234", "unit": "ratio", "explanation": "ratio", "sources": ["pre_0"]}
    validate(good, answer_schema(record))
    with pytest.raises(ValidationError):
        validate({**good, "value": "0." + "1234567890" * 30}, answer_schema(record))


def test_grounded_final_rejects_hallucinated_value_unit_or_sources(record):
    calc = Calculator(record)
    result = calc.calculate("divide", ["table_r1_c1", "table_r1_c2"])
    schema = grounded_answer_schema(record, [result])
    good = {"value": "120", "unit": "percent", "explanation": "ratio", "sources": ["table_r1_c1", "table_r1_c2"]}
    validate(good, schema)
    for patch in ({"value": "12"}, {"unit": "USD"}, {"sources": ["pre_0"]}):
        with pytest.raises(ValidationError):
            validate({**good, **patch}, schema)
    assert decimal_output("13.4000") == "13.4"
    assert decimal_output("-0.00000000000000000000000001") == "0"
