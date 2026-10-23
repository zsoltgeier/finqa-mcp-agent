#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null
uname -a
free -h
df -h .
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
fi
if [ ! -x .conda/envs/finqa-mcp/bin/python ]; then
  conda create -y -p "$PWD/.conda/envs/finqa-mcp" --file conda-linux-64.lock.txt
fi
conda run -p .conda/envs/finqa-mcp python -m pip install -r requirements.txt
conda run -p .conda/envs/finqa-mcp python -m pip install -e . --no-deps --no-build-isolation
if [ ! -x .runtime/ollama/bin/ollama ]; then
  conda run -p .conda/envs/finqa-mcp python scripts/install_ollama.py
fi
conda run -p .conda/envs/finqa-mcp python -m finqa_mcp.cli prepare
echo 'Indítás külön terminálban: bash scripts/start_ollama.sh'
echo 'Utána: OLLAMA_HOST=127.0.0.1:11434 .runtime/ollama/bin/ollama pull qwen3:4b-q4_K_M'
