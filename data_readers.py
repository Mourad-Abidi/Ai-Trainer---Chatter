"""Text extraction for non-plain-text training files (.pdf, .json, .jsonl).

Kept free of torch/numpy so it can be imported and tested on its own.

    extract_text(path, raw_bytes, encoding, errors, json_keys=None) -> str

* ``.txt`` / ``.md`` / anything else  -> decoded as-is.
* ``.json``                           -> every string VALUE in the document
                                         (keys are skipped), see below.
* ``.jsonl`` / ``.ndjson``            -> one JSON value per line, same rule.
* ``.pdf``                            -> text layer of every page.

JSON handling
-------------
Without ``json_keys`` all string values are used, in document order. Pass
``json_keys=["text", "content"]`` to keep ONLY values stored under those keys
(case-insensitive, at any depth) — the right choice for datasets that mix
real text with ids, urls and labels. Numbers, booleans and nulls are ignored.
Each top-level record becomes one block; blocks are separated by a blank line,
strings inside a record by a single newline.

PDF handling
------------
Needs ONE of: ``pypdf`` (recommended), ``PyMuPDF``, ``pdfplumber``::

    pip install pypdf

Only the text layer is read. Scanned (image-only) PDFs contain no text and
need OCR first (e.g. ``ocrmypdf``); they are reported as empty.
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

JSON_SUFFIXES = {".json"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
PDF_SUFFIXES = {".pdf"}
BINARY_SUFFIXES = PDF_SUFFIXES            # formats that must not be .decode()d

_INSTALL_HINT = ("PDF support needs a PDF library. Install one with:  "
                 "pip install pypdf")


# ── JSON ─────────────────────────────────────────────────────────────────
def _collect_strings(node: Any, keys: set[str] | None, out: list[str],
                     under_key: bool = False) -> None:
    """Depth-first walk appending string values to ``out``."""
    if isinstance(node, str):
        if keys is None or under_key:
            s = node.strip()
            if s:
                out.append(s)
    elif isinstance(node, dict):
        for k, v in node.items():
            hit = keys is not None and str(k).lower() in keys
            # Once inside a selected key, keep everything below it.
            _collect_strings(v, keys, out, under_key or hit)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _collect_strings(v, keys, out, under_key)


def _record_text(node: Any, keys: set[str] | None) -> str:
    parts: list[str] = []
    _collect_strings(node, keys, parts)
    return "\n".join(parts)


def _normalise_keys(json_keys: Iterable[str] | None) -> set[str] | None:
    if not json_keys:
        return None
    ks = {str(k).strip().lower() for k in json_keys if str(k).strip()}
    return ks or None


def extract_json(text: str, json_keys: Iterable[str] | None = None) -> str:
    keys = _normalise_keys(json_keys)
    data = json.loads(text)
    if isinstance(data, list):                       # list of records
        blocks = (_record_text(item, keys) for item in data)
    elif isinstance(data, dict) and len(data) == 1 and \
            isinstance(next(iter(data.values())), list):
        blocks = (_record_text(item, keys)           # {"data": [ ... ]}
                  for item in next(iter(data.values())))
    else:
        blocks = [_record_text(data, keys)]
    return "\n\n".join(b for b in blocks if b)


def extract_jsonl(text: str, json_keys: Iterable[str] | None = None) -> str:
    keys = _normalise_keys(json_keys)
    blocks: list[str] = []
    bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            blocks.append(_record_text(json.loads(line), keys))
        except json.JSONDecodeError:
            bad += 1
    if bad and not blocks:
        raise ValueError("not valid JSON Lines (no line parsed)")
    return "\n\n".join(b for b in blocks if b)


# ── PDF ──────────────────────────────────────────────────────────────────
_WS_RUNS = re.compile(r"[ \t\u00a0]+")
_BLANKS = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _clean_pdf_text(text: str) -> str:
    """Tidy typical PDF extraction noise (ligatures, hyphen breaks, runs)."""
    text = unicodedata.normalize("NFKC", text)          # ﬁ -> fi, nbsp -> ' '
    text = text.replace("\x0c", "\n\n").replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t"
                   or unicodedata.category(ch)[0] != "C")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)             # "exam-\nple" -> "example"
    text = _WS_RUNS.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return _BLANKS.sub("\n\n", text).strip()


def _pdf_pages_pypdf(raw: bytes) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader                    # legacy name
    reader = PdfReader(io.BytesIO(raw))
    if getattr(reader, "is_encrypted", False):
        try:
            if not reader.decrypt(""):
                raise ValueError("PDF is password-protected")
        except Exception as exc:                        # noqa: BLE001
            raise ValueError(f"PDF is encrypted ({exc})") from exc
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:                               # noqa: BLE001
            pages.append("")                            # skip one bad page
    return pages


def _pdf_pages_fitz(raw: bytes) -> list[str]:
    import fitz  # PyMuPDF
    with fitz.open(stream=raw, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password-protected")
        return [page.get_text() for page in doc]


def _pdf_pages_plumber(raw: bytes) -> list[str]:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def extract_pdf(raw: bytes) -> str:
    errors: list[str] = []
    found_lib = False
    for reader in (_pdf_pages_pypdf, _pdf_pages_fitz, _pdf_pages_plumber):
        try:
            pages = reader(raw)
        except ImportError:
            continue
        except Exception as exc:                        # noqa: BLE001
            found_lib = True
            errors.append(f"{reader.__name__.split('_')[-1]}: {exc}")
            continue
        return _clean_pdf_text("\n\n".join(pages))
    if not found_lib:
        raise RuntimeError(_INSTALL_HINT)
    raise ValueError("; ".join(errors) or "could not parse PDF")


# ── dispatcher ───────────────────────────────────────────────────────────
def extract_text(path: Path | str, raw: bytes, encoding: str = "utf-8",
                 errors: str = "replace",
                 json_keys: Iterable[str] | None = None) -> str:
    """Return the training text contained in ``raw`` (the file's bytes)."""
    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return extract_pdf(raw)
    enc = "utf-8-sig" if encoding.lower().replace("_", "-") in ("utf-8", "utf8") \
        else encoding
    text = raw.decode(enc, errors=errors)
    if suffix in JSON_SUFFIXES:
        return extract_json(text, json_keys)
    if suffix in JSONL_SUFFIXES:
        return extract_jsonl(text, json_keys)
    return text