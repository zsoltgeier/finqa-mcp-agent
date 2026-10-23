import hashlib
import json

import pytest
from pydantic import ValidationError

from finqa_mcp.cli import load_config
from finqa_mcp.common import canonical_hash, write_json
from finqa_mcp.prepare import clean, download_locked
from finqa_mcp.schemas import InputRecord, load_record


def test_annotations_never_enter_clean_record(tmp_path):
    raw = {"id": "../../outside", "qa": {"question": "How much?", "exe_ans": "SECRET",
           "program": "SECRET", "gold_inds": {"pre_0": "SECRET"}},
           "pre_text": ["Sales"], "post_text": [], "table": [["Year", 2020]]}
    record = clean(raw)
    assert "SECRET" not in record.model_dump_json()
    assert set(record.model_dump()) == {"id", "question", "pre_text", "post_text", "table"}
    write_json(tmp_path / f"{canonical_hash(record.id)}.json", record.model_dump())
    assert load_record(tmp_path, record.id) == record
    with pytest.raises(ValidationError):
        InputRecord.model_validate({**record.model_dump(), "answer": "SECRET"})


def test_download_checksum_and_cache(tmp_path, monkeypatch):
    source = tmp_path / "upstream.json"
    source.write_text('{"value": 42}')
    spec = {"url": source.as_uri(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    target = tmp_path / "cache/download.json"
    assert download_locked(spec, target).read_bytes() == source.read_bytes()
    source.unlink()
    assert download_locked(spec, target) == target
    target.write_text("damaged")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_locked(spec, target)


def test_failed_download_not_installed(tmp_path):
    source = tmp_path / "upstream"
    source.write_text("bad data")
    target = tmp_path / "cache/download.json"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_locked({"url": source.as_uri(), "sha256": "0" * 64}, target)
    assert not target.exists()
    assert not target.with_suffix(".json.part").exists()


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    path = tmp_path / "project/config/app.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"input_dir": "data/inputs", "tokenizer_path": ".runtime/tokenizer.json",
                                "demonstrations_path": "data/demos.json"}))
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config["input_dir"] == str(tmp_path / "project/data/inputs")
    assert config["tokenizer_path"] == str(tmp_path / "project/.runtime/tokenizer.json")
