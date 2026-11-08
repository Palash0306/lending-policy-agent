# Deployment guide

The agent runs entirely on your own machine: Ollama (the language model), Chroma (the document index) and a small web server. No policy text is sent anywhere.

## 1. Install on a company laptop

You need: Python 3.10 or newer, about 10 GB of free disk, and 16 GB of RAM or more (32 GB is comfortable; the 7B model alone uses about 5 GB). A GPU is not needed.

**macOS**
```bash
brew install ollama python@3.12            # or install both from their websites
ollama serve &                              # leave it running
ollama pull qwen2.5:7b
cd lending-policy-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/serve.py                     # then open http://127.0.0.1:8000
```

**Windows (PowerShell)**
```powershell
# Install Ollama from https://ollama.com/download/windows and Python from python.org (tick "Add to PATH")
ollama pull qwen2.5:7b
cd lending-policy-agent
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts\serve.py
```
Check the model first with `python scripts/verify_ollama.py`.

## 2. Networks that block downloads (offline set-up)

Three things are downloaded: Python packages, the Ollama model, and the embedding model (from Hugging Face). On a firewalled network, download them once on a machine with internet and copy them across.

1. **Python packages**: `pip download -r requirements.txt -d wheels`, copy the `wheels` folder, then `pip install --no-index --find-links wheels -r requirements.txt`.
2. **Ollama model**: run `ollama pull qwen2.5:7b` on the open machine, then copy the folder `~/.ollama/models` (macOS/Linux) or `%USERPROFILE%\.ollama\models` (Windows) to the same place on the target. To keep models elsewhere, set the `OLLAMA_MODELS` environment variable to that folder.
3. **Embedding model**: on the open machine run `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`, then copy the Hugging Face cache (`~/.cache/huggingface`, or the folder named by `HF_HOME`) to the target. On the target set:
   ```
   HF_HOME=<folder you copied>
   HF_HUB_OFFLINE=1
   ```
   `HF_HUB_OFFLINE=1` stops it trying to reach the internet on every start.

## 3. Sharing one Ollama server

The model is the heavy part. Several people can share one stronger machine:
- On the server run Ollama with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` (only on a network you trust: Ollama has no login either).
- On each laptop set `OLLAMA_HOST=http://that-server:11434` in `.env`. Policy text then travels only inside your network.
- Requests are queued one at a time per laptop by this tool. Several laptops on one server will queue inside Ollama, so each will be slower.

## 4. Security notes

- **There is no login.** By default the web server listens on `127.0.0.1`, so only the same computer can reach it. Do **not** use `--host 0.0.0.0` on a shared network.
- To let colleagues use one instance, put it behind your company's reverse proxy or single sign-on (for example nginx, an Application Gateway or an identity-aware proxy) and keep the server itself on `127.0.0.1`.
- Uploads are limited to `.pdf` and `.docx` whose first bytes match, with a size limit (`MAX_UPLOAD_MB`). Excel and CSV files never contain formulas: text that starts with `=`, `+`, `-` or `@` is stored as plain text.
- The page sends basic security headers (no framing, no sniffing, no caching, same-origin only).
- Policy documents are internal: `.gitignore` excludes `*.pdf` and `*.docx`, and `scripts/check_no_documents.sh` fails if one is tracked. Enable the pre-commit guard once per clone: `git config core.hooksPath .githooks`.

## 5. Where data lives, and how to delete it

| What | Where | To delete |
|---|---|---|
| Document index | `chroma_db/` (`CHROMA_PERSIST_DIR`) | Stop the server, delete the folder, or use "remove" on the page per document. |
| Uploaded files | `uploads/` (`UPLOAD_DIR`) | Delete files, or "remove" on the page. |
| Batch results | `data/batches/` (`BATCH_DIR`) | "Delete batch" on the page, or delete the batch's folder. |
| Conversations | In memory only | Restart the server. |
| Model and embedding caches | Ollama's and Hugging Face's folders | Delete those folders. |

Nothing is stored outside the project folder except those model caches.

## 6. Settings (`.env`)

See `.env.example` for every setting. The ones you are most likely to change: `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `API_PORT`, `MAX_UPLOAD_MB`, `BATCH_DIR`, `MAX_BATCH_FIELDS`.

## 7. Troubleshooting

| Symptom | What to do |
|---|---|
| "Ollama: not reachable" on the page | Start Ollama (`ollama serve`) and check `OLLAMA_HOST`. |
| "model ... not installed" | `ollama pull qwen2.5:7b` (or the model named in `.env`). |
| A question ends in a timeout error | Extraction takes minutes on a CPU. Raise `OLLAMA_TIMEOUT` (default 300 seconds), for example to 900. |
| Everything is slow | Tick **Quick mode**, close the editor and browser tabs, check `ollama ps` (PROCESSOR should say GPU if you have one), or use a smaller model (`qwen2.5:3b`; check its answers more carefully). |
| The laptop freezes or an editor crashes | Memory pressure. Run the server from a plain terminal, close other programs, use a smaller model. A batch keeps what it finished, so restart it with the remaining fields. |
| Embedding model download fails | Firewall: use the offline steps above. |
| A scanned PDF indexes with "no readable text" | It is an image. OCR is not built; get a text version. |
| Files indexed with `scripts/index_documents.py` do not show on the page | Restart the server; the page reads the list at start-up. |
