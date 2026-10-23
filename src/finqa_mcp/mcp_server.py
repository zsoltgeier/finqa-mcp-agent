import argparse
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from finqa_mcp.search import search_text
from finqa_mcp.schemas import load_record
from finqa_mcp.numbers import Calculator
from finqa_mcp.report import compact_numbers, full_report


class SourceOperand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(description="Exact source ID in the current report, such as table_r1_c2 or post_3")
    number_index: int = Field(default=0, ge=0, description="Zero-based numeric occurrence in text; always 0 for a scalar table cell")


def create_server(input_dir, allowed_id):
    record = load_record(input_dir, allowed_id)
    calculator = Calculator(record)
    server = FastMCP("finqa-mcp")
    @server.tool()
    async def search_report(example_id: str, query: Annotated[str, Field(min_length=1, max_length=1000)],
                            top_k: Annotated[int, Field(ge=1, le=5)] = 3) -> dict:
        """Search current report text; numeric_values contains normalized numbers and number_index for references."""
        if example_id != allowed_id:
            return {"error": "access_denied", "detail": "Only the current example is accessible."}
        output = search_text(record, query, top_k)
        for match in output.get("matches", []):
            match["numeric_values"] = compact_numbers(calculator.numeric_values(match["source_id"]))
        return output

    @server.tool()
    def read_table(example_id: str) -> dict:
        """Read the current table with row labels, year/column labels, original strings and parsed numeric values."""
        if example_id != allowed_id:
            return {"error": "access_denied", "detail": "Only the current example is accessible."}
        return {"example_id": allowed_id, "table": full_report(record)["table"]}

    @server.tool()
    def calculate(operation: Literal["add", "subtract", "multiply", "divide"],
                  operands: Annotated[list[str | SourceOperand], Field(min_length=2, max_length=2)]) -> dict:
        """One operation on two decimals, source IDs, {source_id,number_index} objects or prior calc_N results.
        A mixed-value table cell can be disambiguated using number_index from numeric_values.
        Percent cells become fractional ratios (36.3% -> 0.363); currency and provenance are preserved.
        Returns result_id, value, unit and sources. Reuse result_id for later operations."""
        return calculator.calculate(operation, [x.model_dump() if isinstance(x, SourceOperand) else x for x in operands])

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--example-id", required=True)
    args = parser.parse_args()
    create_server(args.input_dir, args.example_id).run(transport="stdio")


if __name__ == "__main__":
    main()
