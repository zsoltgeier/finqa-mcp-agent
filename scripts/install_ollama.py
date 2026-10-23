"""Install the pinned, checksummed Ollama binary inside the project."""
import platform
import tarfile
from pathlib import Path

from finqa_mcp.common import read_json
from finqa_mcp.prepare import download_locked


def main():
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("The bundled installer supports Linux/WSL x86_64 only.")
    lock = read_json(root / "config/ollama.json")
    archive = download_locked(lock, root / f".runtime/downloads/ollama-{lock['version']}-linux-amd64.tgz")
    destination = root / ".runtime/ollama"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    print(f"Installed Ollama {lock['version']} at {destination}")


if __name__ == "__main__":
    main()
