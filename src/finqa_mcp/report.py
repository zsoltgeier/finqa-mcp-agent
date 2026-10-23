from finqa_mcp.numbers import Calculator


def compact_numbers(numbers):
    return [{k: n[k] for k in ("number_index", "value", "unit", "percent_value") if k in n} for n in numbers]


def full_report(record):
    calc = Calculator(record)
    rows = []
    for r, values in enumerate(record.table):
        rows.append({"row": r, "label": values[0] if values else "", "cells": [
            {"source_id": f"table_r{r}_c{c}", "original": value,
             "numeric_values": compact_numbers(calc.numeric_values(f"table_r{r}_c{c}"))}
            for c, value in enumerate(values)]})
    return {"table": rows, "text": [
        {"source_id": sid, "text": text, "numeric_values": compact_numbers(calc.numeric_values(sid))}
        for sid, text in calc.sources.items() if sid.startswith(("pre_", "post_"))]}
