"""Run the final implementation without any historical experiment artifacts."""
import argparse
import asyncio
import json
from pathlib import Path

import yaml

from finqa_mcp.common import canonical_hash, read_json, write_json
from finqa_mcp.prepare import prepare
from finqa_mcp.runner import run
from finqa_mcp.schemas import InputRecord, load_record


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    for key in ("input_dir", "tokenizer_path", "demonstrations_path"):
        config[key] = str((root / config[key]).resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description="Local FinQA baseline and MCP agent")
    parser.add_argument("--config", type=Path, default=Path("config/app.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Download pinned sources and prepare clean inputs")
    setup.add_argument("--split", nargs="+", choices=("train", "dev", "test"), default=["train"])
    execute = commands.add_parser("run", help="Answer one question and save the complete call log")
    selection = execute.add_mutually_exclusive_group(required=True)
    selection.add_argument("--id", help="FinQA ID from a prepared split")
    selection.add_argument("--input", type=Path, help="Custom clean JSON: id, question, pre_text, post_text, table")
    execute.add_argument("--split", choices=("train", "dev", "test"), default="train")
    execute.add_argument("--variant", choices=("baseline", "agent", "all"), default="agent")
    execute.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.config.resolve().parent.parent
    if args.command == "prepare":
        print(json.dumps(prepare(root, args.split)))
        return
    config = load_config(args.config)
    if args.input:
        record = InputRecord.model_validate(read_json(args.input))
        # The MCP process needs the same validated record, never the raw input.
        directory = args.output_dir.resolve() / "inputs"
        write_json(directory / f"{canonical_hash(record.id)}.json", record.model_dump())
    else:
        directory = Path(config["input_dir"]) / args.split
        record = load_record(directory, args.id)
    config["input_dir"] = str(directory)
    demonstrations = read_json(config["demonstrations_path"])
    variants = ("baseline", "agent") if args.variant == "all" else (args.variant,)
    failed = False
    for variant in variants:
        result = asyncio.run(run(record, variant, config, demonstrations))
        destination = args.output_dir / f"{canonical_hash(record.id)}-{variant}.json"
        write_json(destination, result)
        print(json.dumps({"variant": variant, "status": result["status"],
                          "answer": result["answer"], "log": str(destination)}, ensure_ascii=False))
        failed |= result["status"] != "ok"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
