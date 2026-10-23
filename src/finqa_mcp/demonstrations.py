"""Worked calculations authored on three fixed train records only."""
from finqa_mcp.common import canonical_hash
from finqa_mcp.schemas import source_map
from finqa_mcp.numbers import Calculator

specs = [
    ("ETR/2004/page_22.pdf-1", "percentage_change", [("subtract", ["table_r3_c1", "table_r3_c2"]),
        ("divide", ["calc_1", "table_r3_c2"])], ["table_r0_c1", "table_r0_c2", "table_r3_c0", "table_r3_c1", "table_r3_c2"], "percent"),
    ("RE/2015/page_131.pdf-4", "percentage_share", [("divide", [
        {"source_id": "pre_2", "number_index": 0}, {"source_id": "pre_1", "number_index": 2}])], ["pre_1", "pre_2"], "percent"),
    ("ALLE/2017/page_118.pdf-1", "average", [("add", [{"source_id": "post_15", "number_index": 0},
        {"source_id": "post_15", "number_index": 1}]), ("add", ["calc_1", {"source_id": "post_15", "number_index": 2}]),
        ("divide", ["calc_2", "3"])], ["post_15"], "million_USD")]


def build_demonstrations(train_records):
    records = {record.id: record for record in train_records}
    examples = []
    for identifier, kind, operations, evidence_ids, unit in specs:
        record = records[identifier]
        sources = source_map(record)
        calc, steps = Calculator(record), []
        for op, operands in operations:
            result = calc.calculate(op, operands)
            assert "error" not in result, result
            steps.append({"operation": op, "operands": operands, "result": result})
        final = steps[-1]["result"]
        value = final["percent_value"] if unit == "percent" else final["value"]
        examples.append({"id": identifier, "kind": kind, "question": record.question,
            "source_sha256": canonical_hash(record.model_dump()), "clean_record": record.model_dump(),
            "evidence": {sid: sources[sid] for sid in evidence_ids}, "steps": steps,
            "answer": {"value": value, "unit": unit, "explanation": kind.replace("_", " ") + " using the cited operands and intermediate results.",
                       "sources": final["sources"]}})
    return {"origin": "train only; manually specified source-derived worked calculations",
            "annotation_fields_in_prompt": False, "examples": examples}
