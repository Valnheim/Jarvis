# Terminal Jarvis

A small Python chat loop that talks to a **local Ollama** model with a Jarvis-inspired voice (calm, dry, addresses you as Sir). Memory lasts for this terminal session only.

## Requirements

- Python 3.11+
- [Ollama for Windows](https://ollama.com/download)

## Setup

1. Install Ollama from [ollama.com/download](https://ollama.com/download). **Close and reopen** PowerShell (or Windows Terminal) so `ollama` is on your PATH. Then pull the default chat model — this is a real ~4.9 GB download, not a chat prompt. Run it in **PowerShell**, not in Cursor chat:

   ```powershell
   ollama pull llama3.1:8b
   ```

   If `ollama` is not recognized, use the full path:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.1:8b
   ```

   Confirm it is installed: `ollama list` should show `llama3.1:8b`. Do not use `llama3` as a substitute unless you also pass `--model llama3` to `jarvis.py`.

   For notes search, also pull the embedding model (one time, a few hundred MB):

   ```powershell
   ollama pull nomic-embed-text
   ```

2. From this folder, create a virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running (`ollama serve` if it is not already).

## Run

```bash
python jarvis.py
```

Optional:

```bash
python jarvis.py --model llama3.1:8b
```

Environment overrides: `JARVIS_MODEL`, `JARVIS_HOST` (default `http://127.0.0.1:11434`), `JARVIS_EMBED_MODEL` (default `nomic-embed-text`).

## Notes

On the first launch, Jarvis asks for a path to a `.txt` or `.md` file, or a folder of them. Enter skips this and chat works without notes. It walks subfolders, splits the text into passages, and stores a local search index in `.jarvis/index.json`. Later launches reuse that index.

Each question searches the index and sends the closest passages to the chat model. The model weights are not retrained. Chat history is still cleared when you quit; the notes index stays until you forget it.

## In the terminal

- Type normally to chat.
- `/help` — commands
- `/index` — choose a path and rebuild the notes index
- `/sources` — list indexed files
- `/forget` — delete the notes index
- `/reset` — clear this chat session (keeps the persona and the notes)
- `/exit` — quit (Ctrl+C / Ctrl+D also work)

Edit `persona.md` to change the voice without touching Python.
