"""Pinned downloads and an explicit boundary between raw data and model inputs."""
import urllib.request
from pathlib import Path

from finqa_mcp.common import canonical_hash, digest, read_json, write_json
from finqa_mcp.schemas import InputRecord


def download_locked(spec, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            urllib.request.urlretrieve(spec["url"], temporary)
            if digest(temporary) != spec["sha256"]:
                raise ValueError(f"Checksum mismatch: {destination}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(destination) != spec["sha256"]:
        raise ValueError(f"Checksum mismatch: {destination}; remove the damaged cache and retry.")
    return destination


def clean(row):
    # Copy only the question and source text. Answers, programs, and gold indices
    # stay in the raw download and are never read by the agent or MCP server.
    return InputRecord(id=row["id"], question=row["qa"]["question"],
                       pre_text=row["pre_text"], post_text=row["post_text"],
                       table=[[str(cell) for cell in line] for line in row["table"]])


def prepare(root, splits=("train",)):
    root = Path(root)
    lock = read_json(root / "config/finqa-source.json")
    counts = {}
    for split in dict.fromkeys(("train", *splits)):
        source = download_locked(lock["files"][split], root / f"data/source/{split}.json")
        rows = read_json(source)
        records = [clean(row) for row in rows]
        for record in records:
            write_json(root / f"data/inputs/{split}/{canonical_hash(record.id)}.json", record.model_dump())
        write_json(root / f"data/inputs/{split}/index.json", [r.id for r in records])
        counts[split] = len(records)
        if split == "train":
            from finqa_mcp.demonstrations import build_demonstrations
            write_json(root / "data/demonstrations.json", build_demonstrations(records))
    download_locked(read_json(root / "config/tokenizer.json"), root / ".runtime/tokenizer/tokenizer.json")
    return counts
