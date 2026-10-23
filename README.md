# FinQA MCP Agent

Helyi pénzügyi kérdésmegválaszoló Qwen3-4B modellel és Ollamával. Két futási mód: teljes forrásból válaszoló `baseline`, illetve MCP-keresést, táblázatolvasást és számolást használó `agent`. A számoló cellahivatkozásokat, köztes eredményeket és mértékegységeket is kezel.

**Környezet:** Linux/WSL2 x86_64, conda; GPU-s futtatáshoz elérhető NVIDIA GPU. WSL-ben a Windows NVIDIA-driver szükséges. A telepítő projektkönyvtáron belüli Python-környezetet és Ollamát hoz létre, rögzített verziókkal és ellenőrzött letöltésekkel.

Telepítés a repó gyökeréből:

```bash
bash scripts/setup.sh
bash scripts/start_ollama.sh
```

A szerver fusson tovább; egy másik terminálban:

```bash
OLLAMA_HOST=127.0.0.1:11434 .runtime/ollama/bin/ollama pull qwen3:4b-q4_K_M
conda run -p .conda/envs/finqa-mcp python -m finqa_mcp.cli run \
  --id 'PNC/2018/page_45.pdf-2' --variant all
```

A `--variant all` mindkét módot lefuttatja; külön is választható a `baseline` vagy az `agent`. Saját tisztított rekord: `--input kerdes.json`, mezői: `id`, `question`, `pre_text`, `post_text`, `table`. A szövegmezők sztringlisták, a táblázat sztringekből álló sorok listája.

Beállítások: [`config/app.yaml`](config/app.yaml). A FinQA-adatok és a tokenizer automatikusan letöltődnek; a modell bemenetébe kizárólag a kérdés és a források kerülnek. További adatrész: `python -m finqa_mcp.cli prepare --split dev test` a conda-környezetben, majd futtatáskor `--split dev` vagy `--split test`. A válaszok és hívásnaplók az `outputs/` könyvtárba kerülnek; azonos kérdés és mód ismétlése felülírja a korábbi naplót.

Tesztek, Ollama és adatletöltés nélkül:

```bash
conda run -p .conda/envs/finqa-mcp python -m pytest -q
```

A repó a forráskódot, teszteket és környezetleíró fájlokat tartalmazza. A letöltött adatok, modellek, naplók és kísérleti anyagok Gitből kizártak. Adatforrás: [FinQA](https://github.com/czyssrs/FinQA), modell: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B); felhasználásukra a források saját feltételei vonatkoznak.
