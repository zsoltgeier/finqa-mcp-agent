import json
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from jsonschema import validate
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class Bridge:
    def __init__(self, session, definitions):
        self.session = session
        self.definitions = {tool.name: tool for tool in definitions}
        self.tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                       "parameters": t.inputSchema}} for t in definitions]

    async def _call(self, name, arguments):
        self.last_dispatched = False
        if name not in self.definitions:
            return {"error": "unknown_tool"}
        try:
            validate(arguments, self.definitions[name].inputSchema)
        except Exception as exc:
            return {"error": "invalid_arguments", "detail": str(exc).splitlines()[0]}
        self.last_dispatched = True
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            return {"error": "mcp_error", "detail": " ".join(getattr(c, "text", "") for c in result.content)}
        if result.structuredContent is not None:
            return result.structuredContent
        return json.loads("".join(getattr(c, "text", "") for c in result.content))


    async def call(self, name, arguments):
        output = await self._call(name, arguments)
        if "error" in output:
            output.setdefault("retry_hint", "Change the arguments, inspect numeric_values, or use another source; do not repeat this call.")
        return output


@asynccontextmanager
async def connect(input_dir, example_id):
    params = StdioServerParameters(command=sys.executable, args=["-m", "finqa_mcp.mcp_server",
                                   "--input-dir", str(input_dir), "--example-id", example_id],
                                   env={"PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=15)) as session:
            await session.initialize()
            definitions = (await session.list_tools()).tools
            if {t.name for t in definitions} != {"search_report", "read_table", "calculate"}:
                raise RuntimeError("Unexpected MCP tool inventory")
            yield Bridge(session, definitions)


def inline_schema(schema):
    definitions = schema.get("$defs", {})

    def walk(value):
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            if "$ref" in value:
                return walk(definitions[value["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in value.items() if k not in {"$defs", "title"}}
        return value
    return walk(schema)


def action_schema(tools, answer_schema, allow_final=True):
    branches = []
    for tool in tools:
        fn = tool["function"]
        branches.append({"type": "object", "properties": {
            "action": {"type": "string", "enum": ["tool"]},
            "name": {"type": "string", "enum": [fn["name"]]},
            "arguments": inline_schema(fn["parameters"])},
            "required": ["action", "name", "arguments"], "additionalProperties": False})
    if allow_final:
        branches.append({"type": "object", "properties": {
            "action": {"type": "string", "enum": ["final"]}, "answer": answer_schema},
            "required": ["action", "answer"], "additionalProperties": False})
    return {"oneOf": branches}
