"""Index local .txt/.md notes and search them with Ollama embeddings."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / ".jarvis" / "index.json"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
NOTE_SUFFIXES = {".txt", ".md"}
CHUNK_CHARS = 1500
OVERLAP_CHARS = 200
MAX_FILE_CHARS = 200_000
EMBED_BATCH = 16
TOP_K = 4


class EmbedModelMissing(RuntimeError):
    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(model)


@dataclass
class Index:
    source_path: str
    files: list[str]
    chunks: list[dict]

    def to_json(self) -> dict:
        return {
            "source_path": self.source_path,
            "files": self.files,
            "chunks": self.chunks,
        }


def embed_model_name() -> str:
    return os.environ.get("JARVIS_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def load_index() -> Index | None:
    if not INDEX_PATH.is_file():
        return None
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        chunks = data["chunks"]
        files = data["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"Could not read {INDEX_PATH}: {exc}")
        return None
    return Index(
        source_path=str(data.get("source_path") or ""),
        files=[str(item) for item in files],
        chunks=list(chunks),
    )


def save_index(index: Index) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index.to_json(), ensure_ascii=False),
        encoding="utf-8",
    )


def forget_index() -> bool:
    if not INDEX_PATH.is_file():
        return False
    INDEX_PATH.unlink()
    return True


def note_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() in NOTE_SUFFIXES:
            return [path]
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in filenames:
            if Path(name).suffix.lower() in NOTE_SUFFIXES:
                found.append(Path(dirpath) / name)
    return sorted(found)


def chunk_text(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _split_long(paragraph):
            if not current:
                current = piece
                continue
            combined = f"{current}\n\n{piece}"
            if len(combined) <= CHUNK_CHARS:
                current = combined
                continue
            chunks.append(current)
            tail = current[-OVERLAP_CHARS:].strip()
            current = f"{tail}\n\n{piece}" if tail else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph) <= CHUNK_CHARS:
        return [paragraph]
    pieces: list[str] = []
    start = 0
    while start < len(paragraph):
        end = min(start + CHUNK_CHARS, len(paragraph))
        pieces.append(paragraph[start:end].strip())
        if end >= len(paragraph):
            break
        start = end - OVERLAP_CHARS
    return [piece for piece in pieces if piece]


def _read_note(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        print(f"Truncated {path.name} at {MAX_FILE_CHARS} characters.")
        text = text[:MAX_FILE_CHARS]
    return text


def _display_source(root: Path, path: Path) -> str:
    if root.is_file():
        return path.name
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def embed_texts(
    client: httpx.Client,
    host: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    response = client.post(
        f"{host}/api/embed",
        json={"model": model, "input": texts},
    )
    if response.status_code >= 400:
        body = response.text
        if response.status_code == 404 or "not found" in body.lower():
            raise EmbedModelMissing(model)
        raise RuntimeError(f"Ollama embed HTTP {response.status_code}: {body.strip()}")
    payload = response.json()
    if err := payload.get("error"):
        if "not found" in str(err).lower():
            raise EmbedModelMissing(model)
        raise RuntimeError(str(err))
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding payload.")
    return embeddings


def build_index(
    client: httpx.Client,
    host: str,
    model: str,
    raw_path: str,
) -> Index:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    files = note_files(path)
    if not files:
        raise ValueError(f"No .txt or .md files in {path}")

    pending: list[dict] = []
    for file_path in files:
        try:
            text = _read_note(file_path)
        except OSError as exc:
            print(f"Skipped {file_path}: {exc}")
            continue
        source = _display_source(path, file_path)
        for passage in chunk_text(text):
            pending.append({"source": source, "text": passage})
    if not pending:
        raise ValueError(f"No readable text in {path}")

    print(f"Indexing {len(files)} file(s)...")
    vectors: list[list[float]] = []
    texts = [item["text"] for item in pending]
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors.extend(embed_texts(client, host, model, batch))
    chunks = [
        {"source": item["source"], "text": item["text"], "embedding": vector}
        for item, vector in zip(pending, vectors, strict=True)
    ]
    index = Index(
        source_path=str(path),
        files=sorted({item["source"] for item in chunks}),
        chunks=chunks,
    )
    save_index(index)
    return index


def search(
    index: Index,
    client: httpx.Client,
    host: str,
    model: str,
    query: str,
    k: int = TOP_K,
) -> list[dict]:
    if not index.chunks or not query.strip():
        return []
    vector = embed_texts(client, host, model, [query])[0]
    ranked = sorted(
        index.chunks,
        key=lambda item: _cosine(vector, item.get("embedding") or []),
        reverse=True,
    )
    hits = []
    for item in ranked[:k]:
        hits.append({"source": item.get("source") or "", "text": item.get("text") or ""})
    return [hit for hit in hits if hit["text"]]


def format_notes(hits: list[dict]) -> str:
    if not hits:
        return ""
    blocks = ["Notes from the user's files:"]
    for hit in hits:
        blocks.append(f"[{hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
