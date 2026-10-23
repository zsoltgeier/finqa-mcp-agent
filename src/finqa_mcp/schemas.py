from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Decimal
    unit: Literal["number", "percent", "percentage_point", "ratio", "USD", "thousand_USD", "million_USD", "billion_USD", "years"]
    explanation: str = Field(min_length=1, max_length=1600)
    sources: list[str] = Field(min_length=1, max_length=30)

    @field_validator("value")
    @classmethod
    def finite(cls, value):
        if not value.is_finite():
            raise ValueError("value must be finite")
        return value


def source_map(record: InputRecord):
    sources = {}
    for side in ("pre", "post"):
        sources.update({f"{side}_{i}": text for i, text in enumerate(getattr(record, f"{side}_text"))})
    for r, row in enumerate(record.table):
        for c, value in enumerate(row):
            sources[f"table_r{r}_c{c}"] = value
    return sources


def load_record(directory, example_id):
    # The filename is a hash: IDs containing slashes cannot escape the input root.
    from finqa_mcp.common import canonical_hash, read_json
    from pathlib import Path
    record = InputRecord.model_validate(read_json(Path(directory) / f"{canonical_hash(example_id)}.json"))
    if record.id != example_id:
        raise ValueError("ID mismatch")
    return record
