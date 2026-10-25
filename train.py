# train.py
"""Train a character-level GRU on a text corpus.

Usage
-----
    python train.py                  # open the training launcher GUI (default)
    python train.py --cli            # headless CLI training
    python train.py --gui            # same as default (backwards compat)
    python train.py --gui --device cpu --steps 5000   # GUI pre-filled
    python train.py --cli --device cuda:0 --num-threads 4
    python train.py --cli --fresh    # ignore the saved model, start from zero

Training is CUMULATIVE. If a saved model already exists, the next session
continues from it (all it learned is kept; new characters in new data are
added to its vocabulary). Your old model is only replaced when the new one
scores better on the validation text, and it is backed up first whenever
you explicitly start from scratch (see checkpoint_backups/).

Data can be supplied three ways (they combine):

    --data FILE_OR_DIR_OR_GLOB [...]   # files, dirs, or globs
    --data-list MANIFEST.txt           # one path per line (repeatable)
    (nothing)                          # auto-detect (see below)

Auto-detect, when no data is given, tries in order: ``data_to_train_on/``
(next to this script, then in the current directory), then 1.txt..10.txt in
the current directory, then 1.txt..10.txt next to this script.

The checkpoint path is anchored to the folder that holds this script unless
you pass an absolute path, so train.py and ask.py always agree on where it
lives no matter which directory you launch them from.

Directories and glob matches are walked recursively, picking up files whose
extension is in ``--data-ext`` (default ``.txt,.text,.md,.json,.jsonl,.pdf``).
JSON / JSONL files contribute their string values (restrict with
``--json-keys text,content``); PDFs contribute their text layer (needs
``pip install pypdf``). For corpora with
thousands of files prefer ``--data-list`` — it dodges the OS command-line
length limit entirely.

When training finishes successfully, a streaming chat window opens
automatically unless --no-auto-chat is passed.
"""
from __future__ import annotations

import argparse
import codecs
import contextlib
import csv
import glob
import math
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from data_readers import BINARY_SUFFIXES, extract_text
from model import (
    CharGRU,
    EMA,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATA_DIRNAME,
    PROJECT_DIR,
    atomic_torch_save,
    describe_checkpoint,
    find_checkpoint,
    load_state_expanding_vocab,
    load_torch_file,
    resolve_checkpoint_path,
)

try:
    import numpy as _np
except ImportError:
    _np = None

# ─────────────────────────────────────────────────────────────────────────
#  LOSS-REDUCTION PLAYBOOK  (read before touching hyperparameters)
# ─────────────────────────────────────────────────────────────────────────
#  1. MORE DATA.      A ~2M-param GRU saturates around 5-10M characters.
#                     If val loss plateaus high, add text first.
#  2. BIGGER BLOCK.   --block-size 512 exposes longer dependencies.
#                     Halve --batch-size to keep memory flat.
#  3. LABEL SMOOTHING --label-smoothing 0.05 often gives a small val win.
#  4. WEIGHT DECAY    0.01 safe, 0.05-0.1 for very small corpora.
#  5. LOWER PEAK LR   Loss spikes → halve --learning-rate.
#  6. TIE WEIGHTS     --tie-weights (needs embedding_dim==hidden_size).
#  7. DROPOUT         0.1-0.3 for small data, lower for large.
#  8. UNIGRAM INIT    On by default — free warmup savings.
#  9. KEEP TRAINING   Sessions accumulate: every new session continues from
#                     the saved model, so "a bit more training" is never lost.
# ─────────────────────────────────────────────────────────────────────────

BASE_CHARACTERS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    "&'@#.-()*=+/,?!:;[]{}\n"
)
# step         = CUMULATIVE step of the model (continues across sessions)
# train_loss   = running average while training (dropout on, label smoothing on)
# train_clean  = loss on a fixed slice of the training text, measured exactly
#                like validation (no dropout, averaged weights) - the honest
#                number to compare with val_loss when judging over-fitting
# session      = 1 for a model's first training session, 2 for the next, ...
LOG_COLUMNS = ["run_id", "step", "elapsed", "train_loss", "val_loss",
               "lr", "grad_norm", "tokens_per_sec", "train_clean", "session"]
_LEGACY_LOG_COLUMNS = LOG_COLUMNS[:8]
DEFAULT_DATA_EXTS = (".txt", ".text", ".md", ".json", ".jsonl", ".pdf")
DEFAULT_DATA_EXTS_STR = ",".join(DEFAULT_DATA_EXTS)


# ─────────────────────────────────────────────────────────────────────────
#  Shared helpers (used by both CLI and GUI)
# ─────────────────────────────────────────────────────────────────────────
def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,),
                           device=data.device)
    idx = starts[:, None] + torch.arange(block_size, device=data.device)[None, :]
    return data[idx], data[idx + 1]


def prepare_log(path: Path):
    """Open the CSV log for appending. Older logs are upgraded in place so the
    history (and the dashboard's continuous curves) is never thrown away."""
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != LOG_COLUMNS:
            if header == _LEGACY_LOG_COLUMNS:
                try:
                    with open(path, newline="") as f:
                        rows = list(csv.reader(f))[1:]
                    tmp = path.with_name(path.name + ".tmp")
                    with open(tmp, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(LOG_COLUMNS)
                        for r in rows:
                            w.writerow(r + [""] * (len(LOG_COLUMNS) - len(r)))
                    os.replace(tmp, path)
                    print(f"Upgraded old log format: {path.name}")
                except OSError as exc:
                    print(f"warning: could not upgrade {path.name} ({exc})")
            else:
                backup = path.with_name(
                    f"{path.stem}_old_{int(time.time())}{path.suffix}")
                path.rename(backup)
                print(f"Old-format log moved to {backup}")
    is_new = not path.exists() or path.stat().st_size == 0
    handle = open(path, "a", newline="")
    writer = csv.writer(handle)
    if is_new:
        writer.writerow(LOG_COLUMNS)
        handle.flush()
    return handle, writer


def launch_live_plot(log_path: Path, run_id: str):
    """Open the live dashboard. It shows every session of this model so the
    curves continue where the previous session stopped."""
    script = Path(__file__).resolve().with_name("plot_results.py")
    try:
        return subprocess.Popen(
            [sys.executable, str(script), "--live",
             "--log", str(log_path.resolve())]
        )
    except Exception as exc:
        print(f"(Could not open live plot window: {exc})")
        return None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto" or not device_arg:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"Requested '{device_arg}' but CUDA is not available. "
                "Try --device cpu or install a CUDA-enabled torch."
            )
        return torch.device(device_arg)
    return torch.device(device_arg)


def configure_runtime(device: torch.device, num_threads: int) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
        print(f"Torch CPU threads: {num_threads}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.set_device(idx)
        print(f"CUDA device: {torch.cuda.get_device_name(idx)}")


def _can_open_gui() -> bool:
    try:
        import tkinter as tk
        root = tk.Tk(); root.destroy()
        return True
    except Exception:
        return False


def _maybe_launch_chat(checkpoint: Path) -> None:
    if not _can_open_gui():
        print("(Chat window skipped: no display. "
              "Run `python ask.py --cli` to talk to the model.)")
        return
    script = Path(__file__).resolve().with_name("ask.py")
    try:
        subprocess.Popen([sys.executable, str(script), "--gui",
                          "--checkpoint", str(checkpoint)])
        print("Chat window opened. Close it when you're done.")
    except Exception as exc:
        print(f"(Could not open chat window: {exc})")


def _shell_join(cmd: list[str]) -> str:
    """Quote a command line for display so it can be pasted into a shell."""
    if sys.platform == "win32":
        return subprocess.list2cmdline(cmd)
    import shlex
    return shlex.join(cmd)


def _fmt_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fixed_windows(data: torch.Tensor, block_size: int, n: int):
    """Evenly spaced, DETERMINISTIC windows over ``data``.

    Random validation batches make the curve jitter and let a lucky draw win
    "best checkpoint". Fixed windows mean every evaluation (and every
    session) is measured on exactly the same text, so numbers are comparable.
    """
    last = data.numel() - block_size - 2          # largest legal start
    n = max(1, min(int(n), last + 1))
    starts = torch.linspace(0, last, n).round().long().to(data.device)
    idx = starts[:, None] + torch.arange(block_size, device=data.device)[None, :]
    return data[idx], data[idx + 1]


def load_previous_checkpoint(path: Path) -> dict | None:
    """Load the saved model at ``path`` for continued training.

    Returns the checkpoint dict (tensors on CPU) or ``None`` if no file
    exists. Raises ``RuntimeError`` when a file exists but cannot be used.
    The weights are test-loaded into a throw-away model so every problem is
    discovered *before* any training time is spent.
    """
    found = find_checkpoint(path, allow_fallback=False)
    if found is None:
        return None
    ckpt = load_torch_file(found, "cpu")
    need = ("characters", "embedding_dim", "hidden_size", "model_state_dict")
    missing = [k for k in need if k not in ckpt]
    if missing:
        raise RuntimeError(f"file lacks {missing} (made by an older train.py?)")
    probe = CharGRU(
        len(ckpt["characters"]), int(ckpt["embedding_dim"]),
        int(ckpt["hidden_size"]), int(ckpt.get("num_layers", 2)), 0.0,
        tie_weights=bool(ckpt.get("tie_weights", False)))
    load_state_expanding_vocab(
        probe, ckpt["model_state_dict"], len(ckpt["characters"]))
    ckpt["_path"] = str(found)
    return ckpt


def backup_checkpoint(path: Path) -> Path | None:
    """Copy ``path`` into ``checkpoint_backups/`` (never lose a trained model)."""
    try:
        folder = path.parent / "checkpoint_backups"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"
        shutil.copy2(path, dest)
        return dest
    except OSError as exc:
        print(f"warning: could not back up {path}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────
#  Data-spec expansion helpers — scale to thousands of files
# ─────────────────────────────────────────────────────────────────────────
def _parse_extensions(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated extension list into a normalized tuple.

    Empty string → ``()``, meaning *accept any file*.
    """
    if not spec or not spec.strip():
        return ()
    out: list[str] = []
    for part in spec.split(","):
        s = part.strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        out.append(s)
    return tuple(out)


def _walk_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively find files under ``root`` matching ``extensions``.

    Skips any path component starting with ``.`` (``.git``, ``__pycache__``,
    editor swap files). Pass such files explicitly via ``--data`` if you
    really want them included.
    """
    exts = set(extensions) if extensions else None
    out: list[Path] = []
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if exts is not None and p.suffix.lower() not in exts:
                continue
            out.append(p)
    except OSError as exc:
        print(f"warning: could not fully scan {root}: {exc}")
    out.sort()
    return out


def expand_data_specs(
    specs: list[str], extensions: tuple[str, ...], *, verbose: bool = True,
) -> list[Path]:
    """Resolve files, directories, and globs into a sorted, de-duplicated list.

    * Glob (contains ``*?[``) → expanded recursively; directory matches walked.
    * Directory              → walked recursively for ``extensions`` files.
    * File                   → added directly.

    Missing specs are collected and reported together rather than aborting.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    missing: list[str] = []

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except OSError:
            rp = p.absolute()
        if not rp.is_file() or rp in seen:
            return
        seen.add(rp)
        found.append(rp)

    def _add_dir(d: Path) -> None:
        for f in _walk_files(d, extensions):
            _add(f)

    for spec in specs:
        spec = str(spec).strip()
        if not spec:
            continue
        if any(ch in spec for ch in "*?["):
            matches = glob.glob(spec, recursive=True)
            if not matches:
                missing.append(spec)
                continue
            for m in sorted(matches):
                mp = Path(m)
                if mp.is_dir():
                    _add_dir(mp)
                else:
                    _add(mp)
            continue
        p = Path(spec)
        if p.is_dir():
            _add_dir(p)
        elif p.is_file():
            _add(p)
        else:
            missing.append(spec)

    if missing and verbose:
        preview = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
        print(f"warning: {len(missing)} data spec(s) not found: {preview}")
    return found


def read_data_manifest(manifest_path: Path) -> list[str]:
    """Read a manifest: one path per line, ``#`` comments and blanks skipped.

    Relative paths resolve **against the manifest's parent directory** —
    so a manifest is a self-contained, portable file list.
    """
    base = manifest_path.parent
    out: list[str] = []
    text = manifest_path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            p = base / p
        out.append(str(p))
    return out


def load_corpus(
    files: list[Path],
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    separator: str = "\n\n",
    verbose: bool = True,
    json_keys: list[str] | None = None,
) -> tuple[str, dict[str, int], int]:
    """Read every file; return ``(joined_text, char_counts, replacements)``.

    Uses a list of pieces + one ``join`` — linear in total size, unlike
    ``text += piece`` in a loop (which is O(files²) in bytes copied).
    Per-file decode errors become a *warning*, not a crash.
    """
    pieces: list[str] = []
    counts: Counter = Counter()
    replacements = 0
    total_bytes = 0
    empty: list[Path] = []
    unreadable: list[tuple[Path, str]] = []
    t0 = time.time()
    log_every = max(1, len(files) // 20)  # ~5% progress steps

    for i, path in enumerate(files, 1):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            unreadable.append((path, str(exc)))
            continue
        total_bytes += len(raw)
        try:
            # .txt/.md decode as text; .json/.jsonl yield their string
            # values; .pdf yields its text layer (see data_readers.py).
            text = extract_text(path, raw, encoding, errors, json_keys)
        except (LookupError, UnicodeError, ValueError, RuntimeError,
                ImportError) as exc:
            unreadable.append((path, str(exc)))
            continue
        if "\ufffd" in text and path.suffix.lower() not in BINARY_SUFFIXES:
            replacements += text.count("\ufffd")
        text = text.strip()
        if not text:
            empty.append(path)
            continue
        pieces.append(text)
        counts.update(text)

        if verbose and (i % log_every == 0 or i == len(files)):
            print(f"  read {i:,}/{len(files):,} files "
                  f"({total_bytes/1024/1024:.1f} MiB, "
                  f"{sum(counts.values()):,} chars)")

    # The '\n\n' separators between files are part of the final text —
    # fold them into the unigram counts so the output-bias init is exact.
    if len(pieces) > 1:
        for ch in separator:
            counts[ch] += len(pieces) - 1

    if unreadable:
        print(f"warning: {len(unreadable)} file(s) could not be read:")
        for path, msg in unreadable[:5]:
            print(f"  {path}: {msg}")
        if len(unreadable) > 5:
            print(f"  … and {len(unreadable) - 5} more")
    if empty:
        print(f"note: {len(empty)} file(s) were empty after stripping.")
        if any(p.suffix.lower() == ".pdf" for p in empty):
            print("  (a PDF with no text is usually a scan — run OCR first, "
                  "e.g. `ocrmypdf in.pdf out.pdf`)")
    if replacements:
        print(f"warning: {replacements:,} byte(s) undecodable as {encoding!r} "
              f"→ U+FFFD. Use --data-encoding to change the codec.")

    joined = separator.join(pieces)
    if verbose:
        print(f"  loaded {len(files):,} file(s) → {len(joined):,} chars "
              f"({len(counts):,} distinct) in {time.time() - t0:.1f}s")
    return joined, dict(counts), replacements


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Map each character to an int id. Numpy fast path when available."""
    if not text:
        return torch.empty(0, dtype=torch.long)
    if _np is not None:
        arr = _np.fromiter(map(stoi.__getitem__, text),
                           dtype=_np.int64, count=len(text))
        return torch.from_numpy(arr)
    return torch.tensor(list(map(stoi.__getitem__, text)), dtype=torch.long)


def _default_data_specs(extensions: tuple[str, ...]) -> list[str]:
    """Find training data when the user passed none.

    Order: ``data_to_train_on/`` (script dir, then CWD), then ``1.txt..10.txt``
    in the CWD, then next to this script. The first location that actually
    contains matching files wins; the choice is printed so it is never magic.
    """
    cwd = Path.cwd()
    folders = [PROJECT_DIR / DEFAULT_DATA_DIRNAME, cwd / DEFAULT_DATA_DIRNAME]
    seen: set[Path] = set()
    for folder in folders:
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        if _walk_files(folder, extensions):
            print(f"No --data given: using folder {folder}")
            return [str(folder)]
    for base in (cwd, PROJECT_DIR):
        numbered = [str(base / f"{i}.txt") for i in range(1, 11)
                    if (base / f"{i}.txt").is_file()]
        if numbered:
            print(f"No --data given: using {len(numbered)} numbered file(s) "
                  f"in {base}")
            return numbered
    return []


# ─────────────────────────────────────────────────────────────────────────
#  Argument validation
# ─────────────────────────────────────────────────────────────────────────
def _validate_training_args(args: argparse.Namespace) -> None:
    """Reject obviously-broken configurations before doing any work."""
    if args.steps < 1:
        raise SystemExit("error: --steps must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("error: --batch-size must be >= 1")
    if args.block_size < 8:
        raise SystemExit("error: --block-size must be >= 8")
    if args.grad_accum < 1:
        raise SystemExit("error: --grad-accum must be >= 1")
    for name in ("eval_interval", "log_interval", "eval_batches",
                 "embedding_dim", "hidden_size", "num_layers"):
        if getattr(args, name) < 1:
            raise SystemExit(
                f"error: --{name.replace('_', '-')} must be >= 1")
    if args.patience < 0 or args.sample_every < 0 or args.warmup_steps < 0:
        raise SystemExit(
            "error: --patience, --sample-every and --warmup-steps must be >= 0")
    if args.learning_rate <= 0:
        raise SystemExit("error: --learning-rate must be > 0")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise SystemExit("error: --min-lr-ratio must be in [0, 1]")
    if not (0.0 <= args.dropout < 1.0):
        raise SystemExit("error: --dropout must be in [0, 1)")
    if args.weight_decay < 0:
        raise SystemExit("error: --weight-decay must be >= 0")
    if args.label_smoothing < 0:
        raise SystemExit("error: --label-smoothing must be >= 0")
    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("error: --val-fraction must be in (0, 1)")
    if args.tie_weights and args.embedding_dim != args.hidden_size:
        raise SystemExit("error: --tie-weights requires "
                         "--embedding-dim == --hidden-size")
    if not 0.0 <= args.ema_decay < 1.0:
        raise SystemExit("error: --ema-decay must be in [0, 1) "
                         "(0 disables weight averaging)")
    if args.grad_clip < 0:
        raise SystemExit("error: --grad-clip must be >= 0 (0 disables)")
    if args.resume_lr_scale <= 0:
        raise SystemExit("error: --resume-lr-scale must be > 0")

    # Data-specific validation
    try:
        codecs.lookup(args.data_encoding)
    except LookupError:
        raise SystemExit(
            f"error: unknown --data-encoding {args.data_encoding!r}")
    if args.data_errors not in ("strict", "replace", "ignore"):
        raise SystemExit(
            f"error: --data-errors must be strict/replace/ignore, "
            f"got {args.data_errors!r}")
    for manifest in (args.data_list or []):
        if not Path(manifest).exists():
            raise SystemExit(f"error: --data-list file not found: {manifest}")


# ─────────────────────────────────────────────────────────────────────────
#  Training (used by CLI and by the GUI-spawned subprocess)
# ─────────────────────────────────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> int:
    _validate_training_args(args)

    # Anchor the checkpoint to the project folder (not the shell's CWD) so
    # ask.py can always find it. See model.resolve_checkpoint_path.
    args.checkpoint = str(resolve_checkpoint_path(args.checkpoint))
    print(f"Checkpoint will be saved to: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)

    # ── 0. Is there a saved model to CONTINUE from? ──────────────────
    # Sessions are cumulative: unless --fresh is given, a model that already
    # exists is loaded and trained further instead of being replaced.
    resume_mode = "no" if args.fresh else args.resume
    prev: dict | None = None
    if resume_mode != "no":
        try:
            prev = load_previous_checkpoint(ckpt_path)
        except Exception as exc:                     # noqa: BLE001
            if resume_mode == "yes":
                raise SystemExit(
                    f"error: cannot continue the saved model: {exc}")
            print(f"warning: the saved model could not be used ({exc}).\n"
                  f"         Training a fresh model; the old file is backed "
                  f"up first.")
    if resume_mode == "yes" and prev is None:
        raise SystemExit(f"error: --resume yes, but there is no saved model "
                         f"at {ckpt_path}")
    existing = find_checkpoint(ckpt_path, allow_fallback=False)
    if prev is None and existing is not None:
        dest = backup_checkpoint(existing)
        if dest is not None:
            print(f"Starting from zero. Your previous model was backed up "
                  f"to: {dest}")

    if prev is not None:
        # The shape of the network is fixed by the saved weights.
        locked = dict(
            embedding_dim=int(prev["embedding_dim"]),
            hidden_size=int(prev["hidden_size"]),
            num_layers=int(prev.get("num_layers", 2)),
            tie_weights=bool(prev.get("tie_weights", False)))
        changed = [f"{k}: {getattr(args, k)} -> {v}"
                   for k, v in locked.items() if getattr(args, k) != v]
        for k, v in locked.items():
            setattr(args, k, v)
        print("Continuing from your existing model — "
              + describe_checkpoint(prev) + ".")
        print("  Everything it already learned is kept. "
              "(Use --fresh to start from zero instead.)")
        if changed:
            print("  Network size is locked to the saved model: "
                  + ", ".join(changed))
    base_step = int((prev or {}).get("total_steps", (prev or {}).get("step", 0))
                    or 0)
    prev_sessions = int((prev or {}).get("sessions", 1) or 1) if prev else 0
    session_no = prev_sessions + 1
    lineage = list((prev or {}).get("lineage", []))
    # Machine-readable line for the launcher GUI.
    print(f"SESSION base_step={base_step} session={session_no} "
          f"resumed={1 if prev is not None else 0}")

    # ── 1. Resolve specs → concrete file list ────────────────────────
    extensions = _parse_extensions(args.data_ext)

    specs: list[str] = [str(p) for p in (args.data or [])]
    for manifest in (args.data_list or []):
        specs.extend(read_data_manifest(Path(manifest)))

    if not specs:
        specs = _default_data_specs(extensions)
        if not specs:
            raise SystemExit(
                "error: no training data found.\n"
                f"  Looked in: {PROJECT_DIR / DEFAULT_DATA_DIRNAME}, "
                f"{Path.cwd() / DEFAULT_DATA_DIRNAME}, and for 1.txt..10.txt "
                "in the current and script directories.\n"
                f"  Fix: python train.py --cli --data {DEFAULT_DATA_DIRNAME}")

    print(f"Resolving {len(specs):,} data spec(s) "
          f"(extensions: {', '.join(extensions) or 'any/'})…")
    files = expand_data_specs(specs, extensions)
    if not files:
        raise SystemExit("error: data specs resolved to zero files.")
    print(f"  → {len(files):,} file(s)")

    if args.data_shuffle:
        random.Random(args.seed).shuffle(files)
        print(f"  shuffled (seed {args.seed})")
    else:
        files.sort()

    # ── 2. Load corpus ───────────────────────────────────────────────
    print("Reading corpus…")
    all_text, char_counts, _ = load_corpus(
        files, encoding=args.data_encoding, errors=args.data_errors,
        json_keys=[k for k in args.json_keys.split(",") if k.strip()]
        if args.json_keys else None,
    )
    if len(all_text) < 10000:
        raise SystemExit(
            f"error: dataset too small: {len(all_text):,} characters "
            "(need at least 10,000).")

    # ── 3. Vocab + unigram prior (skip the separate bincount pass) ───
    # A continued model keeps its old vocabulary order EXACTLY (so every
    # learned weight still points at the right character) and only appends
    # characters that are new in this data.
    wanted = set(char_counts) | set(BASE_CHARACTERS)
    if prev is not None:
        prev_chars = list(prev["characters"])
        extra = sorted(wanted - set(prev_chars))
        characters = prev_chars + extra
        if extra:
            shown = "".join(extra)[:40]
            print(f"New characters in this data: {len(extra)} ({shown!r}) — "
                  f"added to the model's alphabet; old knowledge is kept.")
    else:
        prev_chars, extra = [], []
        characters = sorted(wanted)
    stoi = {c: i for i, c in enumerate(characters)}
    itos = characters

    counts_t = torch.tensor(
        [char_counts.get(c, 0) for c in characters], dtype=torch.float)
    log_prior = torch.log((counts_t / counts_t.sum()).clamp_min(1e-8))

    # ── 4. Encode (numpy fast path when available) ───────────────────
    n_chars = len(all_text)
    split = int(n_chars * (1 - args.val_fraction))
    if min(split, n_chars - split) <= args.block_size + 1:
        raise SystemExit(
            "error: Train/val split too small for --block-size. "
            "Use more text or a smaller block size.")

    print(f"Encoding {n_chars:,} characters…")
    ids = encode_text(all_text, stoi)
    del all_text  # free before allocating model buffers
    train_data, val_data = ids[:split], ids[split:]

    # ── 5. Train ─────────────────────────────────────────────────────
    # A different seed every session: with a constant seed, session 2 would
    # replay exactly the same random text windows as session 1 and learn
    # nothing new from unchanged data.
    run_seed = args.seed + 7919 * prev_sessions
    torch.manual_seed(run_seed)
    device = resolve_device(args.device)
    configure_runtime(device, args.num_threads)
    train_data, val_data = train_data.to(device), val_data.to(device)

    use_amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16
                 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    _scaler_on = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=_scaler_on)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=_scaler_on)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True   # free speed on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    def autocast():
        return torch.autocast(device_type=device.type, dtype=amp_dtype,
                              enabled=use_amp)

    vocab = len(characters)
    print(f"Device: {device} | mixed precision: "
          f"{amp_dtype if use_amp else 'off'}")
    print(f"Characters: train {train_data.numel():,} | "
          f"val {val_data.numel():,} | vocab {vocab}")
    print(f"Random-guess loss would be ln({vocab}) = {math.log(vocab):.3f}")

    model = CharGRU(
        vocab, args.embedding_dim, args.hidden_size, args.num_layers,
        args.dropout, tie_weights=args.tie_weights,
    ).to(device)
    if prev is not None:
        added = load_state_expanding_vocab(
            model, prev["model_state_dict"], len(prev_chars))
        print("Loaded the previous model's knowledge"
              + (f" (+{added} new characters)" if added else "") + ".")
    elif args.unigram_bias_init:
        model.init_output_bias(log_prior.to(device))
        print("Initialised output bias with unigram prior.")
    print(f"Parameters: {model.num_parameters:,}")

    decay = [q for q in model.parameters() if q.ndim >= 2]
    no_decay = [q for q in model.parameters() if q.ndim < 2]
    # A model that is already trained must not be shaken by a full-size
    # learning rate: continue more gently.
    peak_lr = args.learning_rate * (args.resume_lr_scale
                                    if prev is not None else 1.0)
    if prev is not None and args.resume_lr_scale != 1.0:
        print(f"Learning rate: peak {peak_lr:.2e} "
              f"(x{args.resume_lr_scale:g} because the model is already trained)")
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95),
    )
    min_lr = peak_lr * args.min_lr_ratio

    def lr_at(step: int) -> float:
        if step <= args.warmup_steps:
            return peak_lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(
            1, args.steps - args.warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * \
               (1 + math.cos(math.pi * min(1.0, progress)))

    # Weight averaging: evaluation, samples and checkpoints all use the
    # smoothed weights, which are almost always better than the raw ones.
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"Weight averaging (EMA) on, decay {args.ema_decay} — "
              f"validation, samples and checkpoints use averaged weights.")

    def averaged():
        return ema.average_weights(model) if ema is not None \
            else contextlib.nullcontext()

    # Fixed evaluation windows: the SAME text every time, so the validation
    # curve is smooth, honest and comparable between sessions.
    n_eval = min(args.eval_batches * args.batch_size, 8192)
    val_x, val_y = fixed_windows(val_data, args.block_size, n_eval)
    tr_x, tr_y = fixed_windows(train_data, args.block_size, n_eval)
    print(f"Validation uses {val_x.size(0):,} fixed text windows "
          f"(same every time, so progress is measured fairly).")

    @torch.no_grad()
    def eval_windows(xs: torch.Tensor, ys: torch.Tensor) -> float:
        """Mean cross-entropy per character over fixed windows."""
        model.eval()
        total, count = 0.0, 0
        try:
            with averaged():
                for i in range(0, xs.size(0), args.batch_size):
                    x, y = xs[i:i + args.batch_size], ys[i:i + args.batch_size]
                    with autocast():
                        logits = model(x)
                    total += F.cross_entropy(
                        logits.float().reshape(-1, vocab), y.reshape(-1),
                        reduction="sum").item()
                    count += y.numel()
        finally:
            model.train()
        return total / max(1, count)

    @torch.no_grad()
    def show_sample():
        if val_data.numel() <= 64:
            return
        start = int(torch.randint(0, val_data.numel() - 64, (1,)))
        prompt = val_data[start:start + 64][None, :]
        with averaged():
            out = model.generate(prompt, max_new_tokens=200, temperature=0.8,
                                 top_k=40, top_p=0.95)[0].tolist()
        model.train()
        text = "".join(itos[i] for i in out)
        print("  --- sample (prompt | generated) ---")
        print("  " + (text[:64] + " | " + text[64:]).replace("\n", "\n  "))
        print("  ------------------------------------")

    log_path = Path(args.log)
    handle, writer = prepare_log(log_path)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    saved_any = False

    def save_checkpoint(step_in_session: int, val_loss: float) -> None:
        """Atomic write — a crash mid-save can't corrupt the last good file."""
        nonlocal saved_any
        with averaged():
            weights = {k: t.detach().cpu().clone()
                       for k, t in model.state_dict().items()}
        total = base_step + step_in_session
        record = {"run_id": str(args.run_id), "session": session_no,
                  "steps": int(step_in_session), "best_val": float(val_loss),
                  "data_chars": int(n_chars), "time": int(time.time())}
        atomic_torch_save(
            {
                "model_state_dict": weights,
                "ema_decay": args.ema_decay,
                "characters": itos,
                "block_size": args.block_size,
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "tie_weights": args.tie_weights,
                "best_validation_loss": float(val_loss),
                "step": total,                 # cumulative (shown by ask.py)
                "total_steps": total,
                "sessions": session_no,
                "lineage": (lineage + [record])[-30:],
            },
            args.checkpoint)
        saved_any = True

    viewer = None if args.no_live else launch_live_plot(log_path, args.run_id)

    # Starting point: how good is the model we begin with, measured on THIS
    # validation text? Only a better score replaces it on disk, so a bad
    # session can never make the saved model worse.
    best_val, bad_evals, eval_count = math.inf, 0, 0
    if prev is not None:
        best_val = eval_windows(val_x, val_y)
        start_clean = eval_windows(tr_x, tr_y)
        print(f"The model you are continuing scores val {best_val:.4f} "
              f"(perplexity {math.exp(min(best_val, 20)):.2f}). It will only "
              f"be replaced by something better.")
        writer.writerow([args.run_id, base_step, "0.0", "", f"{best_val:.4f}",
                         "", "", "", f"{start_clean:.4f}", session_no])
        handle.flush()

    stop_file = Path(args.stop_file) if args.stop_file else None
    if stop_file is not None:
        stop_file.unlink(missing_ok=True)

    completed_steps = 0
    diverged = False
    run_loss = torch.zeros((), device=device)
    run_n = 0
    t_start = t_last = time.time()
    model.train()

    try:
        for step in range(1, args.steps + 1):
            if stop_file is not None and stop_file.exists():
                print("Stop requested — wrapping up…")
                break
            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for _micro in range(args.grad_accum):
                x, y = get_batch(train_data, args.batch_size, args.block_size)
                with autocast():
                    logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, vocab), y.reshape(-1),
                    label_smoothing=args.label_smoothing)
                # .detach() accumulation: no .item() → no CPU/GPU sync here.
                step_loss += loss.detach() / args.grad_accum
                scaler.scale(loss / args.grad_accum).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip if args.grad_clip > 0 else math.inf)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            run_loss += step_loss
            run_n += 1
            completed_steps = step

            do_eval = step % args.eval_interval == 0 or step == args.steps
            if step % args.log_interval == 0 or do_eval or step == 1:
                train_loss = (run_loss / run_n).item()
                now = time.time()
                tps = run_n * args.batch_size * args.block_size * args.grad_accum \
                      / max(1e-9, now - t_last)
                run_loss.zero_(); run_n = 0

                if not math.isfinite(train_loss):
                    diverged = True
                    print("Training DIVERGED (loss became NaN/inf). Your saved "
                          "model was not touched. Lower the learning rate "
                          "(try half) and start again.")
                    break

                val_loss_str = clean_str = ""
                if do_eval:
                    v = eval_windows(val_x, val_y)
                    tc = eval_windows(tr_x, tr_y)
                    val_loss_str, clean_str = f"{v:.4f}", f"{tc:.4f}"
                    eval_count += 1
                    print(
                        f"step {step:6d} | train {train_loss:.4f} | val {v:.4f} | "
                        f"ppl {math.exp(min(v, 20)):.2f} | lr {lr:.2e} | "
                        f"{tps:,.0f} chars/s | total {base_step + step:,} | "
                        f"clean {tc:.4f}"
                    )
                    if v < best_val:
                        best_val, bad_evals = v, 0
                        save_checkpoint(step, best_val)
                        print(f"  saved best checkpoint (val {best_val:.4f})")
                    else:
                        bad_evals += 1
                    if args.sample_every and eval_count % args.sample_every == 0:
                        show_sample()

                gn = float(grad_norm)
                writer.writerow(
                    [args.run_id, base_step + step, f"{now - t_start:.1f}",
                     f"{train_loss:.4f}", val_loss_str, f"{lr:.6g}",
                     f"{gn:.4f}" if math.isfinite(gn) else "",
                     f"{tps:.0f}", clean_str, session_no])
                handle.flush()
                t_last = time.time()

                if args.patience and bad_evals >= args.patience:
                    print(f"Early stopping: no val improvement for "
                          f"{bad_evals} evals.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted — wrapping up…")
    finally:
        handle.close()
        if stop_file is not None:
            stop_file.unlink(missing_ok=True)

    # Stopped (Ctrl+C / GUI Stop) before the first validation eval of a
    # FRESH model? Then no checkpoint exists yet and ask.py would report
    # "checkpoint not found". Evaluate once and save what we have.
    if best_val == math.inf and completed_steps > 0 and not diverged:
        try:
            v = eval_windows(val_x, val_y)
            best_val = v
            save_checkpoint(completed_steps, v)
            print(f"  saved checkpoint at step {completed_steps} "
                  f"(val {v:.4f}) — stopped before the first scheduled eval")
        except KeyboardInterrupt:
            print("  second interrupt — no checkpoint written.")

    if best_val < math.inf:
        print(f"Training finished. Best validation loss: {best_val:.4f} "
              f"(perplexity {math.exp(min(best_val, 20)):.2f})")
    else:
        print("Training finished without a validation eval — no checkpoint "
              "saved.")
    if saved_any:
        print(f"Your model now has {base_step + completed_steps:,}+ total "
              f"training steps across {session_no} session(s).")
    elif prev is not None:
        print("This session did not beat the model you started from, so your "
              "saved model was left exactly as it was. Try: more steps, a "
              "lower learning rate, or more/better training text.")

    try:
        from plot_results import save_static
        out_png = str(log_path.resolve().with_name("training_dashboard.png"))
        save_static(str(log_path), out_png)
        print(f"Saved final dashboard: {out_png}")
    except Exception as exc:
        print(f"(Could not save final dashboard: {exc})")

    if diverged:
        return 1
    if best_val >= math.inf:
        print("Hint: increase --steps or lower --eval-interval so at least "
              "one validation eval runs.")
    elif args.auto_chat and saved_any:
        _maybe_launch_chat(Path(args.checkpoint))
    else:
        print("Next: run `python ask.py` to query the model.")

    if viewer is not None and viewer.poll() is None:
        print("The live dashboard window stays open — close it when you're "
              "done.")
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  GUI launcher — shown only when --gui is passed
# ─────────────────────────────────────────────────────────────────────────
GUI_BG, GUI_PANEL, GUI_PANEL2 = "#0d0d14", "#16161f", "#1c1c28"
GUI_BORDER, GUI_TEXT, GUI_DIM = "#2a2a3a", "#e8e8f0", "#8a8aa0"
GUI_ACCENT, GUI_ACCENT_FG, GUI_ERR = "#5eead4", "#0d0d14", "#f87171"
GUI_MONO = ("TkFixedFont", 10)
GUI_SANS = ("TkDefaultFont", 10)

HELP_TITLE: dict[str, str] = {
    "data": "Training text", "browse_dir": "Add a folder", "browse": "Add files",
    "clear_data": "Clear the list", "checkpoint": "Model save file",
    "log": "Training diary", "device": "Device", "num_threads": "CPU threads",
    "steps": "Steps (training length)", "batch_size": "Batch size",
    "block_size": "Block size", "learning_rate": "Learning rate",
    "min_lr_ratio": "Final learning rate", "warmup_steps": "Warm-up",
    "weight_decay": "Weight decay", "label_smoothing": "Label smoothing",
    "grad_accum": "Gradient accumulation", "embedding_dim": "Embedding size",
    "hidden_size": "Hidden size (brain size)", "num_layers": "Layers",
    "dropout": "Dropout", "val_fraction": "Practice-test share",
    "eval_interval": "Test every N steps", "eval_batches": "Test size",
    "sample_every": "Show a sample", "patience": "Patience (auto-stop)",
    "auto_chat": "Auto-open chat", "live_dashboard": "Live graphs",
    "ema_decay": "Smoothed weights (EMA)", "grad_clip": "Gradient clip",
    "json_keys": "JSON keys", "preset": "Presets", "tie": "Tie weights",
    "unigram": "Smart start", "start": "Start", "stop": "Stop",
    "help": "Full guide", "open_dashboard": "Dashboard", "open_chat": "Chat",
    "resume": "Continue the existing model", "resume_lr_scale": "Gentle restart",
    "advanced": "Advanced settings", "recommended": "Recommended preset",
    "model_status": "Your saved model",
    "card_step": "Step (this session)", "card_total": "Total steps (all sessions)",
    "card_session": "Session number", "card_train": "Train loss",
    "card_val": "Validation loss", "card_best": "Best validation loss",
    "card_gap": "Overfit gap", "card_ppl": "Perplexity", "card_bpc": "Bits per character",
    "card_lr": "Learning rate", "card_speed": "Speed", "card_elapsed": "Time elapsed",
}

HELP: dict[str, str] = {
    # ── data & files ─────────────────────────────────────────────────
    "data": "The text your AI learns from — think of it as the books you "
            "give a student. Add .txt, .md, .json or .pdf files, or a whole "
            "folder (sub-folders are included).\n\nMore and better text = a "
            "smarter AI. If you leave this empty, the app looks in the "
            "'data_to_train_on' folder next to the program.",
    "browse_dir": "Pick a whole folder of text files. Everything useful "
                  "inside (including sub-folders) is used. One folder can "
                  "hold thousands of files.",
    "browse": "Pick one or more files. They are ADDED to the list — nothing "
              "you already added is removed. Hold Ctrl/Shift to select "
              "several at once.",
    "clear_data": "Empty the list of training text (does not delete any "
                  "files from your computer).",
    "checkpoint": "The save file for your AI's 'brain'. It is updated "
                  "whenever the AI scores better on its practice test, and "
                  "the Chat window reads the same file.\n\nLeave this alone "
                  "unless you want to keep several different AIs: give each "
                  "its own file name.",
    "log": "A diary (CSV spreadsheet) of the training. The live graphs "
           "are drawn from it, and every new session adds to it so you "
           "can see the whole history.",
    # ── your model ───────────────────────────────────────────────────
    "model_status": "Shows whether a trained AI already exists in the save "
                    "file and how much training it has had.\n\nGREEN = a "
                    "model exists and the next session will build on it.\n"
                    "GREY = nothing saved yet, training starts from zero.",
    "resume": "ON (recommended): the AI keeps everything it already learned "
              "and simply studies more. Each session makes it smarter on top "
              "of the last one. If your new text has letters it has never "
              "seen, they are added automatically.\n\nOFF: throw away its "
              "knowledge and start from zero. Your old model is copied to "
              "the 'checkpoint_backups' folder first, so nothing is lost.\n\n"
              "While ON, the brain size (layers / hidden size / embedding) "
              "is locked to the saved model, because a trained brain "
              "cannot change shape.",
    "resume_lr_scale": "When continuing an already-trained AI, learning "
                       "gently is safer than learning at full strength "
                       "(a big nudge can make it forget). 0.5 means 'half "
                       "strength'.\n\nGood value: 0.3 – 1.0. Use 1.0 if you "
                       "are teaching it something very different.",
    # ── device ───────────────────────────────────────────────────────
    "device": "The part of your computer that does the heavy maths. A "
              "graphics card (CUDA) is typically 10–50× faster than the "
              "normal processor (CPU).\n\n'Auto' picks the best one. Leave "
              "it on Auto if unsure.",
    "num_threads": "How many CPU cores to use. 0 = automatic (best for most "
                   "people). Only change it if your computer becomes "
                   "unresponsive while training — then try a smaller number "
                   "like 2 or 4.",
    # ── length / schedule ────────────────────────────────────────────
    "steps": "How long to train in THIS session. One step = the AI studies "
             "a batch of text and adjusts itself a tiny bit.\n\nRough "
             "guide:\n• 1,500 — quick test (minutes)\n• 8,000 — normal on a "
             "CPU\n• 20,000+ — serious run on a GPU\n\nIf you continue an "
             "existing model, these steps are ADDED to what it already had.",
    "batch_size": "How many text snippets the AI studies at the same time. "
                  "Bigger = steadier learning but needs more memory.\n\nIf "
                  "you see an 'out of memory' error, halve this number.",
    "block_size": "How many letters the AI reads in one snippet — how far "
                  "back it can 'remember' while reading. 256 is a good "
                  "balance. Bigger remembers more but is slower and "
                  "needs more memory.",
    "learning_rate": "How big each learning 'nudge' is. Too small = slow "
                     "learning. Too big = the AI gets confused and the loss "
                     "graph jumps around.\n\nGood value: 0.002. If the "
                     "graph goes wild, halve it.",
    "min_lr_ratio": "Learning slows down gradually during the session. This "
                    "is the final speed as a fraction of the starting "
                    "speed. 0.05 = it ends at 5% — a gentle landing that "
                    "squeezes out the last improvements.",
    "warmup_steps": "At the start the AI begins with tiny nudges and ramps "
                    "up to full speed over this many steps. It prevents "
                    "early chaos. Good value: 100–400.",
    "weight_decay": "A tiny 'forgetting' pressure that stops the AI from "
                    "memorising instead of understanding. 0.01 is safe; "
                    "use 0.05–0.1 when you have very little text.",
    "label_smoothing": "Teaches the AI not to be over-confident. 0 = off. "
                       "0.05 often gives a small improvement when the AI "
                       "starts to overfit. (Test scores are always measured "
                       "without it, so they stay honest.)",
    "grad_accum": "Pretend to have a bigger batch on a small computer by "
                  "collecting N batches before each learning step. 1 = "
                  "normal. Raise it if 'batch size' is limited by memory.",
    # ── architecture ─────────────────────────────────────────────────
    "embedding_dim": "How many numbers describe each letter. Small is fine "
                     "(64). Must equal 'Hidden size' only if you turn on "
                     "'Tie weights'.\n\nLocked while you continue an "
                     "existing model.",
    "hidden_size": "The size of the AI's 'brain' (memory per layer). Bigger "
                   "= can learn more complicated things, but slower and "
                   "hungrier for data. 256 = small, 512 = medium, "
                   "768+ = large.\n\nLocked while you continue an existing "
                   "model.",
    "num_layers": "How many brain layers are stacked. 2 is a strong "
                  "default; 3+ only helps with a lot of text.\n\nLocked "
                  "while you continue an existing model.",
    "dropout": "Randomly 'switches off' part of the brain while studying so "
               "it can't just memorise. 0.1 for lots of text, 0.2–0.3 if "
               "you have very little.",
    # ── validation ───────────────────────────────────────────────────
    "val_fraction": "The AI is tested on text it has NEVER studied — a "
                    "practice exam. This is the share of your text kept "
                    "aside for that (taken from the end). 0.05 = 5%.",
    "eval_interval": "Run the practice exam every N steps. Each exam adds "
                     "one point to the graphs and, if the score is the "
                     "best so far, saves the AI.",
    "eval_batches": "How much text each practice exam uses. Bigger = a more "
                    "trustworthy score but a slightly slower exam. 20 is "
                    "plenty for most cases.",
    "sample_every": "Every N exams, print a short piece of text the AI "
                    "wrote, so you can see with your own eyes how it is "
                    "doing. 0 = never.",
    "patience": "Auto-stop: end the session if the exam score has not "
                "improved for N exams in a row. 0 = never stop early.",
    # ── options ──────────────────────────────────────────────────────
    "auto_chat": "Open the Chat window automatically when training "
                 "finishes, so you can talk to your AI straight away.",
    "live_dashboard": "Open the live graphs window while training. Closing "
                      "it does NOT stop the training.",
    "tie": "A trick for small AIs: reuse the same numbers for reading and "
           "writing letters. Saves memory and often helps. Requires "
           "'Embedding size' = 'Hidden size'.\n\nLocked while you continue "
           "an existing model.",
    "unigram": "Starts the AI already knowing how common each letter is "
               "(e.g. 'e' is common, 'z' is rare) instead of guessing "
               "blindly. A free head-start; keep it on. (Only used when "
               "starting from zero.)",
    # ── advanced ─────────────────────────────────────────────────────
    "ema_decay": "Keeps a slowly-moving AVERAGE of the brain and uses that "
                 "for tests and saving. It is almost always better than "
                 "the raw brain, for free.\n\nGood value: 0.999 for "
                 "5,000+ steps, 0.99 for short runs. 0 = off.",
    "grad_clip": "A safety limit on how violent one learning step may be. "
                 "Prevents blow-ups. 1.0 is a safe default; lower it if "
                 "the 'Gradient norm' graph shows big spikes. 0 = off.",
    "json_keys": "Only for .json/.jsonl files. List the fields that contain "
                 "real text, e.g.  text,content  — everything else (ids, "
                 "links, labels) is ignored. Empty = use all text fields.",
    "preset": "One-click starting points. They only fill in the boxes — "
              "check them, then press Start.\n\n• Quick test: finish in "
              "minutes\n• CPU friendly: sensible without a graphics card\n• "
              "Balanced: good on a mid-range GPU\n• High quality: best "
              "results, needs a strong GPU",
    "recommended": "The preset that suits THIS computer best, guessed from "
                   "whether you have a graphics card.",
    "advanced": "Fine-tuning controls (brain size, learning speed, tests, "
                "safety limits). The defaults and presets are good — you "
                "can ignore this whole section if you are new.",
    # ── buttons ──────────────────────────────────────────────────────
    "start": "Begin training with the settings shown. The AI is saved "
             "automatically whenever it improves.  Shortcut: Ctrl+Enter.",
    "stop": "Stop training safely. The best version so far is already "
            "saved, so nothing is lost.",
    "help": "A plain-English guide to every setting and how to read the "
            "graphs.",
    "open_dashboard": "Open the live graphs window (also works after "
                      "training, to look at the history).",
    "open_chat": "Open the Chat window to talk to the saved AI.",
    # ── live metric cards ────────────────────────────────────────────
    "card_step": "How many learning steps this session has done so far.",
    "card_total": "All the steps this AI has EVER done, across all "
                  "sessions. This number only goes up when you continue "
                  "an existing model — proof that training accumulates.",
    "card_session": "Which training session this is for this AI. #1 = "
                    "first time, #2 = the second time you trained it, "
                    "and so on.",
    "card_train": "The AI's mistakes on the text it is studying. Lower = "
                  "better. It should fall quickly, then slowly.",
    "card_val": "The AI's mistakes on its practice exam (text it has never "
                "studied). THIS is the number that matters. Green = it "
                "improved since the last exam.",
    "card_best": "The best exam score reached so far. This is the version "
                 "that is saved to disk.",
    "card_gap": "Exam score minus study score (measured the same way). Near "
                "0 = healthy. Growing steadily = the AI is memorising "
                "instead of understanding (overfitting): add more text or "
                "raise dropout.",
    "card_ppl": "'How many letters is the AI torn between?' on average. "
                "Lower is better; 1 would be perfect. 3 means it is "
                "choosing between about 3 likely letters each time.",
    "card_bpc": "The same score in 'bits per letter' (how many yes/no "
                "questions are needed to guess the next letter). Lower is "
                "better; English text is ~1.0–1.5 for good models.",
    "card_lr": "How big the learning nudges are right now. It rises during "
               "warm-up, then slowly shrinks.",
    "card_speed": "How many letters per second are being studied. Higher "
                  "= faster training.",
    "card_elapsed": "Time since this session started.",
}


class Tooltip:
    """Hover-tooltip. Optionally opens a full info popup on click.

    ``clickable=False`` is used on widgets that already respond to a click
    (buttons, checkboxes) so their real command isn't shadowed by a modal.
    """

    def __init__(self, widget, text: str, delay: int = 450,
                 clickable: bool = True, title: str = "Help"):
        self.widget, self.text, self.delay = widget, text, delay
        self.title = title
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        if clickable:
            widget.bind("<ButtonPress-1>", self._on_click, add="+")

    def _on_enter(self, _e):
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _e=None):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._hide()

    def _on_click(self, _e):
        self._on_leave()
        try:
            from tkinter import messagebox
            messagebox.showinfo(self.title, self.text)
        except Exception:
            pass

    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.text:
            return
        import tkinter as tk
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tip, text=self.text, justify="left",
            background=GUI_PANEL2, foreground=GUI_TEXT,
            relief="solid", borderwidth=1, font=("TkDefaultFont", 10),
            padx=12, pady=8, wraplength=400).pack()
        self._tip = tip

    def _hide(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def _help_badge(parent, key: str):
    """Create a small '?' badge next to a widget."""
    import tkinter as tk
    label = tk.Label(parent, text=" ? ", bg=GUI_PANEL2, fg=GUI_ACCENT,
                     cursor="hand2", font=(GUI_SANS[0], 9, "bold"),
                     padx=3, pady=0)
    Tooltip(label, HELP.get(key, ""), title=HELP_TITLE.get(key, "Help"))
    return label


def _detect_devices() -> list[tuple[str, str]]:
    entries = [("Auto (CUDA if available)", "auto"),
               ("CPU", "cpu")]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                entries.append((f"cuda:{i} ({name})", f"cuda:{i}"))
    except Exception:
        pass
    return entries


def _device_value_from_display(display: str) -> str:
    if display.startswith("Auto"):
        return "auto"
    if display.startswith("CPU"):
        return "cpu"
    m = re.match(r"cuda:(\d+)", display)
    return f"cuda:{m.group(1)}" if m else "auto"


def launch_training_gui(defaults: dict) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter is not available. Run with --cli for terminal training.")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Could not open GUI ({exc}). Run with --cli for terminal "
              f"training.")
        return
    TrainingLauncher(root, defaults)
    root.mainloop()


PRESETS: dict[str, dict] = {
    "Balanced (default)": dict(
        steps=20000, batch_size=128, block_size=256, hidden_size=512,
        embedding_dim=64, num_layers=2, dropout=0.1, learning_rate=0.002,
        warmup_steps=200, weight_decay=0.01, label_smoothing=0.0,
        eval_interval=250, ema_decay=0.999, grad_clip=1.0),
    "Quick test (minutes)": dict(
        steps=1500, batch_size=64, block_size=128, hidden_size=256,
        embedding_dim=64, num_layers=2, dropout=0.1, learning_rate=0.003,
        warmup_steps=50, weight_decay=0.01, label_smoothing=0.0,
        eval_interval=100, ema_decay=0.99, grad_clip=1.0),
    "CPU friendly": dict(
        steps=8000, batch_size=32, block_size=128, hidden_size=256,
        embedding_dim=64, num_layers=2, dropout=0.1, learning_rate=0.003,
        warmup_steps=100, weight_decay=0.01, label_smoothing=0.0,
        eval_interval=200, ema_decay=0.998, grad_clip=1.0),
    "High quality (GPU)": dict(
        steps=40000, batch_size=128, block_size=384, hidden_size=768,
        embedding_dim=96, num_layers=3, dropout=0.15, learning_rate=0.0015,
        warmup_steps=400, weight_decay=0.05, label_smoothing=0.05,
        eval_interval=500, ema_decay=0.9995, grad_clip=1.0),
}

_STEP_LINE = re.compile(
    r"^\s*step\s+(\d+)\s+\|\s+train\s+([\d.]+)\s+\|\s+val\s+([\d.]+)\s+\|"
    r"\s+ppl\s+(\S+)\s+\|\s+lr\s+(\S+)\s+\|\s+([\d,]+)\s+chars/s"
    r"(?:\s+\|\s+total\s+([\d,]+))?(?:\s+\|\s+clean\s+([\d.]+))?")
_SESSION_LINE = re.compile(
    r"^SESSION base_step=(\d+) session=(\d+) resumed=(\d)")


def _fmt_num(v: float) -> str:
    """Compact human number: 1234 -> 1.23k, 2.5e6 -> 2.5M."""
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v / div:.3g}{suf}"
    return f"{v:.3g}"


class TrainingLauncher:
    def __init__(self, root, defaults: dict):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        self.defaults = defaults
        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()
        self.total_steps = int(defaults.get("steps", 20000))
        self.train_start_ts: float | None = None
        self._temp_manifest: Path | None = None
        self.run_id: str | None = None
        self._last_csv_poll = 0.0
        self._best_val = math.inf
        self._prev_val = math.inf
        self.step_offset = 0
        self._stop_file: Path | None = None
        self._arch_locked = False
        self._existing: dict | None = None

        self._apply_theme()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── theme ────────────────────────────────────────────────────────
    def _apply_theme(self):
        tk, ttk = self.tk, self.ttk
        self.root.configure(bg=GUI_BG)
        self.root.title("Ai-Trainer---chatter")
        _h = min(880, max(640, self.root.winfo_screenheight() - 90))
        _w = min(1360, max(1120, self.root.winfo_screenwidth() - 40))
        self.root.geometry(f"{_w}x{_h}")
        self.root.minsize(1100, 620)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=GUI_BG, foreground=GUI_TEXT,
                        fieldbackground=GUI_PANEL, bordercolor=GUI_BORDER,
                        lightcolor=GUI_BORDER, darkcolor=GUI_BORDER)
        style.configure("TFrame", background=GUI_BG)
        style.configure("TLabel", background=GUI_BG, foreground=GUI_TEXT)
        style.configure("Dim.TLabel", background=GUI_BG, foreground=GUI_DIM)
        style.configure("Section.TLabel", background=GUI_BG,
                        foreground=GUI_ACCENT,
                        font=(GUI_SANS[0], 9, "bold"))
        style.configure("TEntry", fieldbackground=GUI_PANEL,
                        foreground=GUI_TEXT, insertcolor=GUI_ACCENT,
                        borderwidth=0)
        style.map("TEntry", foreground=[("disabled", GUI_DIM)],
                  fieldbackground=[("disabled", GUI_BG)])
        style.configure("TCombobox", fieldbackground=GUI_PANEL,
                        background=GUI_PANEL, foreground=GUI_TEXT,
                        arrowcolor=GUI_ACCENT, borderwidth=0)
        style.map("TCombobox", fieldbackground=[("readonly", GUI_PANEL)],
                  foreground=[("readonly", GUI_TEXT)])
        style.configure("TCheckbutton", background=GUI_BG, foreground=GUI_TEXT)
        style.map("TCheckbutton", background=[("active", GUI_BG)])
        style.configure("TButton", background=GUI_PANEL2, foreground=GUI_TEXT,
                        borderwidth=0, padding=(10, 6))
        style.map("TButton", background=[("active", GUI_PANEL)])
        style.configure("Accent.TButton", background=GUI_ACCENT,
                        foreground=GUI_ACCENT_FG, borderwidth=0,
                        font=(GUI_SANS[0], 10, "bold"), padding=(16, 9))
        style.map("Accent.TButton",
                  background=[("active", "#4cd6c4"), ("disabled", GUI_PANEL2)],
                  foreground=[("disabled", GUI_DIM)])
        style.configure("Danger.TButton", background=GUI_PANEL2,
                        foreground=GUI_ERR, borderwidth=0, padding=(10, 6))
        style.configure("TScrollbar", background=GUI_PANEL2,
                        troughcolor=GUI_BG, bordercolor=GUI_BG,
                        arrowcolor=GUI_DIM, borderwidth=0)
        style.configure("TProgressbar", background=GUI_ACCENT,
                        troughcolor=GUI_PANEL, borderwidth=0,
                        lightcolor=GUI_ACCENT, darkcolor=GUI_ACCENT)

    # ── layout ───────────────────────────────────────────────────────
    def _build(self):
        tk, ttk = self.tk, self.ttk

        header = tk.Frame(self.root, bg=GUI_PANEL, height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="◆  Training Launcher", bg=GUI_PANEL,
                 fg=GUI_ACCENT, font=(GUI_SANS[0], 12, "bold"),
                 padx=16).pack(side=tk.LEFT, pady=14)
        tk.Label(header,
                 text="hover or click any  ?  for a plain-English explanation",
                 bg=GUI_PANEL, fg=GUI_DIM, font=(GUI_SANS[0], 9),
                 padx=16).pack(side=tk.RIGHT)
        tk.Frame(self.root, bg=GUI_BORDER, height=1).pack(fill=tk.X)

        body = tk.Frame(self.root, bg=GUI_BG)
        body.pack(fill=tk.BOTH, expand=True)
        left = tk.Frame(body, bg=GUI_BG, width=640)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Frame(body, bg=GUI_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)
        right = tk.Frame(body, bg=GUI_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Action bar pinned to the bottom of the settings pane so Start/Stop
        # are always visible no matter how far the form is scrolled.
        bar = tk.Frame(left, bg=GUI_PANEL)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(bar, bg=GUI_BORDER, height=1).pack(fill=tk.X)
        ctrl = tk.Frame(bar, bg=GUI_PANEL)
        ctrl.pack(fill=tk.X, padx=16, pady=10)

        # Preset bar
        pbar = tk.Frame(left, bg=GUI_BG)
        pbar.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(12, 0))
        tk.Label(pbar, text="Preset", bg=GUI_BG, fg=GUI_TEXT).pack(
            side=tk.LEFT)
        _help_badge(pbar, "preset").pack(side=tk.LEFT, padx=(2, 10))
        self.preset_var = tk.StringVar(value="Select a preset…")
        preset_combo = ttk.Combobox(
            pbar, textvariable=self.preset_var, values=list(PRESETS),
            state="readonly", width=22)
        preset_combo.pack(side=tk.LEFT)
        preset_combo.bind("<<ComboboxSelected>>",
                          lambda _e: self._apply_preset())
        rec = self._recommended_preset()
        rec_btn = ttk.Button(
            pbar, text=f"★ Use recommended ({rec.split(' (')[0]})",
            command=lambda: self._apply_named_preset(rec))
        rec_btn.pack(side=tk.LEFT, padx=(10, 0))
        Tooltip(rec_btn, HELP["recommended"], clickable=False)

        canvas = tk.Canvas(left, bg=GUI_BG, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        form = tk.Frame(canvas, bg=GUI_BG)
        form.bind("<Configure>",
                  lambda _e: canvas.configure(
                      scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=form, anchor="nw")
        # Make the form track the canvas width so wide entries stretch.
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                    padx=(16, 4), pady=8)

        def _wheel(event):
            if getattr(event, "num", 0) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", 0) == 5:
                canvas.yview_scroll(3, "units")
            else:
                canvas.yview_scroll(int(-event.delta / 40) or
                                    (-1 if event.delta > 0 else 1), "units")

        def _over_canvas(event):
            try:
                w = self.root.winfo_containing(event.x_root, event.y_root)
            except (KeyError, self.tk.TclError):  # combobox popdown quirk
                return False
            while w is not None:
                if w is canvas:
                    return True
                w = getattr(w, "master", None)
            return False

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(
                seq, lambda e: _wheel(e) if _over_canvas(e) else None,
                add="+")
        self.root.bind("<Control-Return>", lambda _e: self._start())

        self.vars: dict[str, tk.StringVar] = {}
        self.entries: dict[str, ttk.Entry] = {}

        def field(parent, row, label, key, default, width=14, col=0,
                  colspan=1, padx=(0, 28)):
            tk.Label(parent, text=label, bg=GUI_BG, fg=GUI_TEXT).grid(
                row=row, column=col, sticky="w", pady=3)
            badge = _help_badge(parent, key)
            badge.grid(row=row, column=col + 1, sticky="w", padx=(2, 6))
            var = tk.StringVar(value=str(self.defaults.get(key, default)))
            self.vars[key] = var
            entry = ttk.Entry(parent, textvariable=var, width=width)
            entry.grid(row=row, column=col + 2, columnspan=colspan,
                       sticky="w", padx=padx, pady=3)
            self.entries[key] = entry

        def pair(parent, row, l1, k1, d1, l2, k2, d2):
            field(parent, row, l1, k1, d1, col=0)
            field(parent, row, l2, k2, d2, col=3)

        def section(parent, row, text, first=False):
            ttk.Label(parent, text=text, style="Section.TLabel").grid(
                row=row, column=0, columnspan=6, sticky="w",
                pady=(4, 6) if first else (16, 6))

        # ── DATA ─────────────────────────────────────────────────
        section(form, 0, "DATA & OUTPUT", first=True)

        # Row 1: label + ? + entry (spans columns 2-6)
        tk.Label(form, text="Training text", bg=GUI_BG, fg=GUI_TEXT).grid(
            row=1, column=0, sticky="w", pady=3)
        _help_badge(form, "data").grid(row=1, column=1, sticky="w",
                                       padx=(2, 6))
        self.data_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.data_var).grid(
            row=1, column=2, columnspan=5, sticky="we", pady=3)

        # Seed the field from CLI args. Manifest contents are inlined so the
        # field is self-contained even if the manifest later moves.
        merged: list[str] = [str(p) for p in (self.defaults.get("data") or [])]
        for mpath in (self.defaults.get("data_list") or []):
            mp = Path(mpath)
            if mp.exists():
                try:
                    merged.extend(read_data_manifest(mp))
                except OSError as exc:
                    print(f"warning: could not read manifest {mp}: {exc}")
        self._set_data_paths([p for p in merged if p])

        # Row 2: hint (left) + button bar (right)
        tk.Label(
            form,
            text="Files, folders or globs. Empty = the 'data_to_train_on' folder.",
            bg=GUI_BG, fg=GUI_DIM).grid(
            row=2, column=2, columnspan=2, sticky="w")

        btn_bar = tk.Frame(form, bg=GUI_BG)
        btn_bar.grid(row=2, column=4, columnspan=3, sticky="e",
                     pady=(0, 3))

        browse_files = ttk.Button(btn_bar, text="Add files…",
                                  command=self._browse_data)
        browse_files.pack(side=tk.LEFT, padx=(0, 4))
        Tooltip(browse_files, HELP["browse"], clickable=False)

        browse_dir = ttk.Button(btn_bar, text="Add folder…",
                                command=self._browse_dir)
        browse_dir.pack(side=tk.LEFT, padx=(0, 4))
        Tooltip(browse_dir, HELP["browse_dir"], clickable=False)

        clear_btn = ttk.Button(btn_bar, text="Clear",
                               command=self._clear_data)
        clear_btn.pack(side=tk.LEFT)
        Tooltip(clear_btn, HELP["clear_data"], clickable=False)

        field(form, 3, "Model file", "checkpoint", DEFAULT_CHECKPOINT_NAME,
              width=62, col=0, colspan=3, padx=(0, 0))
        field(form, 4, "Log file", "log", "training_log.csv",
              width=62, col=0, colspan=3, padx=(0, 0))
        self.entries["checkpoint"].bind(
            "<FocusOut>", lambda _e: self._refresh_model_info(), add="+")
        self.entries["checkpoint"].bind(
            "<Return>", lambda _e: self._refresh_model_info(), add="+")

        # ── YOUR MODEL (does the next session continue or restart?) ───
        section(form, 5, "YOUR MODEL")
        status_row = tk.Frame(form, bg=GUI_PANEL, highlightthickness=1,
                              highlightbackground=GUI_BORDER)
        status_row.grid(row=6, column=0, columnspan=6, sticky="we",
                        pady=(0, 6))
        self.model_status = tk.Label(
            status_row, text="", bg=GUI_PANEL, fg=GUI_TEXT, anchor="w",
            justify="left", wraplength=520, padx=12, pady=8,
            font=(GUI_SANS[0], 10))
        self.model_status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        _help_badge(status_row, "model_status").pack(
            side=tk.RIGHT, padx=8, pady=8)

        resume_row = tk.Frame(form, bg=GUI_BG)
        resume_row.grid(row=7, column=0, columnspan=6, sticky="w", pady=2)
        self.opt_resume = tk.BooleanVar(
            value=not bool(self.defaults.get("fresh", False))
            and self.defaults.get("resume", "auto") != "no")
        self.resume_cb = ttk.Checkbutton(
            resume_row, text="Continue training the existing model "
                             "(keep everything it learned)",
            variable=self.opt_resume, command=self._refresh_model_info)
        self.resume_cb.pack(side=tk.LEFT)
        _help_badge(resume_row, "resume").pack(side=tk.LEFT, padx=(4, 0))

        # ── DEVICE ───────────────────────────────────────────────
        section(form, 8, "DEVICE")

        tk.Label(form, text="Device", bg=GUI_BG, fg=GUI_TEXT).grid(
            row=9, column=0, sticky="w", pady=3)
        _help_badge(form, "device").grid(row=9, column=1, sticky="w",
                                        padx=(2, 6))
        self._device_entries = _detect_devices()
        self._device_displays = [d for d, _ in self._device_entries]
        init_display = self._device_displays[0]
        for disp, val in self._device_entries:
            if val == self.defaults.get("device", "auto"):
                init_display = disp
                break
        self.device_var = tk.StringVar(value=init_display)
        self.device_combo = ttk.Combobox(
            form, textvariable=self.device_var, values=self._device_displays,
            state="readonly", width=34)
        self.device_combo.grid(row=9, column=2, sticky="w", padx=(0, 28),
                               pady=3)

        tk.Label(form, text="CPU threads", bg=GUI_BG, fg=GUI_TEXT).grid(
            row=9, column=3, sticky="w", pady=3)
        _help_badge(form, "num_threads").grid(
            row=9, column=4, sticky="w", padx=(2, 6))
        self.vars["num_threads"] = tk.StringVar(
            value=str(self.defaults.get("num_threads", 0) or 0))
        ttk.Entry(form, textvariable=self.vars["num_threads"], width=8).grid(
            row=9, column=5, sticky="w", pady=3)

        # ── TRAINING LENGTH ──────────────────────────────────────
        section(form, 10, "TRAINING LENGTH")
        field(form, 11, "Steps", "steps", 20000, col=0)
        self.total_hint = tk.Label(form, text="", bg=GUI_BG, fg=GUI_DIM,
                                   anchor="w", justify="left")
        self.total_hint.grid(row=11, column=3, columnspan=3, sticky="w")

        # ── OPTIONS ──────────────────────────────────────────────
        section(form, 12, "OPTIONS")

        opt_row = tk.Frame(form, bg=GUI_BG)
        opt_row.grid(row=13, column=0, columnspan=6, sticky="w", pady=4)

        def opt(parent, text, var, key, padx=(0, 4)):
            cb = ttk.Checkbutton(parent, text=text, variable=var)
            cb.pack(side=tk.LEFT, padx=padx)
            _help_badge(parent, key).pack(side=tk.LEFT, padx=(0, 22))
            Tooltip(cb, HELP[key], clickable=False)
            return cb

        self.opt_auto_chat = tk.BooleanVar(
            value=bool(self.defaults.get("auto_chat", True)))
        self.opt_live = tk.BooleanVar(
            value=not self.defaults.get("no_live", False))
        self.opt_tie = tk.BooleanVar(
            value=bool(self.defaults.get("tie_weights", False)))
        self.opt_unigram = tk.BooleanVar(
            value=bool(self.defaults.get("unigram_bias_init", True)))

        opt(opt_row, "Auto-open chat", self.opt_auto_chat, "auto_chat")
        opt(opt_row, "Live graphs", self.opt_live, "live_dashboard")
        self.tie_cb = opt(opt_row, "Tie weights", self.opt_tie, "tie")
        opt(opt_row, "Smart start", self.opt_unigram, "unigram",
            padx=(0, 4))

        # ── ADVANCED (collapsed by default) ──────────────────────
        adv_row = tk.Frame(form, bg=GUI_BG)
        adv_row.grid(row=14, column=0, columnspan=6, sticky="w",
                     pady=(18, 4))
        self.adv_btn = ttk.Button(adv_row, text="▸  Show advanced settings",
                                  command=self._toggle_advanced)
        self.adv_btn.pack(side=tk.LEFT)
        _help_badge(adv_row, "advanced").pack(side=tk.LEFT, padx=(6, 0))

        adv = tk.Frame(form, bg=GUI_BG)
        adv.grid(row=15, column=0, columnspan=6, sticky="we")
        adv.grid_remove()
        self.adv = adv
        self._adv_open = False

        section(adv, 0, "MODEL ARCHITECTURE", first=True)
        pair(adv, 1, "Embedding size", "embedding_dim", 64,
             "Hidden size", "hidden_size", 512)
        pair(adv, 2, "Layers", "num_layers", 2,
             "Dropout", "dropout", 0.1)
        section(adv, 3, "OPTIMIZATION")
        pair(adv, 4, "Batch size", "batch_size", 128,
             "Block size", "block_size", 256)
        pair(adv, 5, "Learning rate", "learning_rate", 0.002,
             "Final LR ratio", "min_lr_ratio", 0.05)
        pair(adv, 6, "Warm-up steps", "warmup_steps", 200,
             "Gentle restart", "resume_lr_scale", 0.5)
        pair(adv, 7, "Weight decay", "weight_decay", 0.01,
             "Label smoothing", "label_smoothing", 0.0)
        pair(adv, 8, "Test share", "val_fraction", 0.05,
             "Grad accum", "grad_accum", 1)
        pair(adv, 9, "Test every", "eval_interval", 250,
             "Test size", "eval_batches", 20)
        section(adv, 10, "SAFETY & EXTRAS")
        pair(adv, 11, "Sample every", "sample_every", 4,
             "Patience", "patience", 0)
        pair(adv, 12, "EMA decay", "ema_decay", 0.999,
             "Grad clip", "grad_clip", 1.0)
        field(adv, 13, "JSON keys", "json_keys", "", width=34, col=0,
              colspan=3, padx=(0, 0))
        adv.columnconfigure(5, weight=1)

        # ── CONTROLS ─────────────────────────────────────────────
        self._ctrl = ctrl
        self.start_btn = ttk.Button(ctrl, text="▶  Start training",
                                    style="Accent.TButton",
                                    command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        _help_badge(ctrl, "start").pack(side=tk.LEFT, padx=(4, 20))
        Tooltip(self.start_btn, HELP["start"], clickable=False)

        self.stop_btn = ttk.Button(ctrl, text="■  Stop",
                                   style="Danger.TButton",
                                   command=lambda: self._stop(silent=False),
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        _help_badge(ctrl, "stop").pack(side=tk.LEFT, padx=(4, 20))
        Tooltip(self.stop_btn, HELP["stop"], clickable=False)

        help_btn = ttk.Button(ctrl, text="?  Guide",
                              command=self._show_help)
        help_btn.pack(side=tk.RIGHT)
        Tooltip(help_btn, HELP["help"], clickable=False)
        chat_btn = ttk.Button(ctrl, text="Chat", command=self._open_chat)
        chat_btn.pack(side=tk.RIGHT, padx=(0, 6))
        Tooltip(chat_btn, HELP["open_chat"], clickable=False)
        dash_btn = ttk.Button(ctrl, text="Dashboard",
                              command=self._open_dashboard)
        dash_btn.pack(side=tk.RIGHT, padx=(0, 6))
        Tooltip(dash_btn, HELP["open_dashboard"], clickable=False)

        self._build_monitor(right)
        form.columnconfigure(5, weight=1)
        self._refresh_model_info()

    def _toggle_advanced(self):
        self._adv_open = not self._adv_open
        if self._adv_open:
            self.adv.grid()
            self.adv_btn.configure(text="▾  Hide advanced settings")
        else:
            self.adv.grid_remove()
            self.adv_btn.configure(text="▸  Show advanced settings")

    @staticmethod
    def _recommended_preset() -> str:
        try:
            if torch.cuda.is_available():
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                return "High quality (GPU)" if gb >= 10 else \
                    "Balanced (default)"
        except Exception:
            pass
        return "CPU friendly"

    def _refresh_model_info(self):
        """Tell the user, in plain words, whether the next session will
        continue an existing model, and lock the brain size if so."""
        tk = self.tk
        info = None
        try:
            path = resolve_checkpoint_path(
                self.vars["checkpoint"].get().strip() or None)
            found = find_checkpoint(path, allow_fallback=False)
            if found is not None:
                ckpt = load_torch_file(found, "cpu")
                if all(k in ckpt for k in
                       ("characters", "embedding_dim", "hidden_size")):
                    info = ckpt
        except Exception:
            info = None
        self._existing = info
        continuing = info is not None and bool(self.opt_resume.get())

        if info is None:
            text, color = ("No saved model yet — the first session starts "
                           "from zero. After that, every new session "
                           "automatically continues from it.", GUI_DIM)
            hint = ""
        elif continuing:
            text, color = (f"✔ Saved model found — {describe_checkpoint(info)}."
                           f"\nThis session will CONTINUE from it: nothing it "
                           f"learned is lost.", "#86efac")
            hint = ("adds on top of the "
                    f"{int(info.get('total_steps', info.get('step', 0)) or 0):,}"
                    " steps it already has")
        else:
            text, color = (f"⚠ Saved model found — {describe_checkpoint(info)}."
                           f"\nYou chose to start from zero. The old model is "
                           f"copied to 'checkpoint_backups' first.", "#fbbf24")
            hint = ""
        self.model_status.configure(text=text, fg=color)
        self.total_hint.configure(text=hint)

        # A trained brain cannot change shape: show and lock its size.
        self._arch_locked = continuing
        if continuing:
            self.vars["embedding_dim"].set(str(int(info["embedding_dim"])))
            self.vars["hidden_size"].set(str(int(info["hidden_size"])))
            self.vars["num_layers"].set(str(int(info.get("num_layers", 2))))
            self.opt_tie.set(bool(info.get("tie_weights", False)))
        state = tk.DISABLED if continuing else tk.NORMAL
        for key in ("embedding_dim", "hidden_size", "num_layers"):
            try:
                self.entries[key].configure(state=state)
            except Exception:
                pass
        try:
            self.tie_cb.configure(state=state)
        except Exception:
            pass

    # ── live monitor (right pane) ────────────────────────────────────
    def _build_monitor(self, parent):
        tk, ttk = self.tk, self.ttk
        wrap = tk.Frame(parent, bg=GUI_BG)
        wrap.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        ttk.Label(wrap, text="LIVE METRICS  (click a ? to learn what it means)",
                  style="Section.TLabel").pack(anchor="w", pady=(4, 6))

        grid = tk.Frame(wrap, bg=GUI_BG)
        grid.pack(fill=tk.X)
        self.cards: dict[str, tk.Label] = {}
        specs = [("step", "STEP (SESSION)", "card_step"),
                 ("total", "TOTAL STEPS", "card_total"),
                 ("session", "SESSION #", "card_session"),
                 ("train", "TRAIN LOSS", "card_train"),
                 ("val", "VAL LOSS", "card_val"),
                 ("best", "BEST VAL", "card_best"),
                 ("gap", "OVERFIT GAP", "card_gap"),
                 ("ppl", "PERPLEXITY", "card_ppl"),
                 ("bpc", "BITS / CHAR", "card_bpc"),
                 ("lr", "LEARNING RATE", "card_lr"),
                 ("speed", "CHARS / SEC", "card_speed"),
                 ("elapsed", "ELAPSED", "card_elapsed")]
        for i, (key, title, hkey) in enumerate(specs):
            card = tk.Frame(grid, bg=GUI_PANEL, highlightthickness=1,
                            highlightbackground=GUI_BORDER)
            card.grid(row=i // 4, column=i % 4, sticky="nsew", padx=4, pady=4)
            top = tk.Frame(card, bg=GUI_PANEL)
            top.pack(fill=tk.X, padx=(12, 6), pady=(8, 0))
            tk.Label(top, text=title, bg=GUI_PANEL, fg=GUI_DIM,
                     font=(GUI_SANS[0], 8, "bold")).pack(side=tk.LEFT)
            _help_badge(top, hkey).pack(side=tk.RIGHT)
            val = tk.Label(card, text="—", bg=GUI_PANEL, fg=GUI_TEXT,
                           font=(GUI_SANS[0], 16, "bold"))
            val.pack(anchor="w", padx=12, pady=(0, 10))
            self.cards[key] = val
        for c in range(4):
            grid.columnconfigure(c, weight=1, uniform="card")

        self.progress = ttk.Progressbar(wrap, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(14, 4))
        self.status_label = ttk.Label(wrap, text="Idle", style="Dim.TLabel")
        self.status_label.pack(anchor="w")

        ttk.Label(wrap, text="TRAINING LOG", style="Section.TLabel").pack(
            anchor="w", pady=(16, 6))
        log_frame = tk.Frame(wrap, bg=GUI_PANEL, highlightthickness=1,
                             highlightbackground=GUI_BORDER)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, bg=GUI_PANEL, fg=GUI_TEXT,
                                font=GUI_MONO, relief=tk.FLAT, wrap=tk.NONE,
                                state=tk.DISABLED, padx=10, pady=8)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal",
                                command=self.log_text.xview)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text.configure(yscrollcommand=log_scroll.set,
                                xscrollcommand=xscroll.set)
        for tag, color in (("cmd", GUI_DIM), ("err", GUI_ERR),
                           ("warn", "#fbbf24"), ("ok", "#86efac"),
                           ("metric", GUI_ACCENT)):
            self.log_text.tag_configure(tag, foreground=color)

    def _reset_cards(self):
        for lbl in self.cards.values():
            lbl.configure(text="—", fg=GUI_TEXT)
        self._best_val = math.inf
        self._prev_val = math.inf
        self.step_offset = 0

    def _update_metrics(self, line: str):
        m = _STEP_LINE.match(line)
        if not m:
            return
        step, train, val, ppl, lr, tps, total, clean = m.groups()
        try:
            v = float(val)
        except ValueError:
            return
        self.cards["step"].configure(text=f"{int(step):,}")
        if total:
            self.cards["total"].configure(text=total)
        self.cards["train"].configure(text=train)
        improved = v < self._prev_val
        self.cards["val"].configure(
            text=val, fg="#86efac" if improved else "#fbbf24")
        self._prev_val = v
        if v < self._best_val:
            self._best_val = v
        self.cards["best"].configure(text=f"{self._best_val:.4f}",
                                     fg=GUI_ACCENT)
        if clean:
            gap = v - float(clean)
            rel = gap / max(v, 1e-9)
            colour = ("#86efac" if rel < 0.10 else
                      "#fbbf24" if rel < 0.25 else GUI_ERR)
            self.cards["gap"].configure(text=f"{gap:+.3f}", fg=colour)
        self.cards["ppl"].configure(text=ppl)
        self.cards["bpc"].configure(text=f"{v / math.log(2):.3f}")
        self.cards["lr"].configure(text=lr)
        self.cards["speed"].configure(text=tps)

    def _log_file(self) -> Path:
        p = Path(self.vars["log"].get().strip() or "training_log.csv")
        p = p.expanduser()
        return p if p.is_absolute() else Path(__file__).resolve().parent / p

    def _poll_csv(self):
        """Smooth progress: read the newest CSV row (written every
        ``--log-interval`` steps) instead of waiting for the next eval."""
        now = time.time()
        if self.run_id is None or now - self._last_csv_poll < 1.0:
            return
        self._last_csv_poll = now
        try:
            path = self._log_file()
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            for row in reversed(tail.splitlines()):
                parts = row.split(",")
                if len(parts) >= 8 and parts[0] == self.run_id:
                    gstep = int(parts[1])               # cumulative step
                    step = max(0, gstep - self.step_offset)
                    self._set_progress(step)
                    self.cards["step"].configure(text=f"{step:,}")
                    self.cards["total"].configure(text=f"{gstep:,}")
                    if parts[3]:
                        self.cards["train"].configure(text=parts[3])
                    if parts[5]:
                        self.cards["lr"].configure(text=parts[5])
                    if parts[7]:
                        self.cards["speed"].configure(
                            text=_fmt_num(float(parts[7])))
                    break
        except (OSError, ValueError, IndexError):
            pass

    def _set_progress(self, step: int):
        if self.total_steps <= 0:
            return
        pct = min(100.0, 100.0 * step / self.total_steps)
        self.progress["value"] = pct
        elapsed = (time.time() - self.train_start_ts
                   if self.train_start_ts else 0.0)
        rate = step / elapsed if elapsed > 0 else 0.0
        remaining = ((self.total_steps - step) / rate
                     if rate > 0 else float("inf"))
        self.cards["elapsed"].configure(text=_fmt_hms(elapsed))
        self.status_label.configure(
            text=f"Step {step:,}/{self.total_steps:,}  ·  {pct:.1f}%  ·  "
                 f"elapsed {_fmt_hms(elapsed)}  ·  ETA {_fmt_hms(remaining)}")
        try:
            self.root.title(f"Ai-Trainer---chatter — {pct:.0f}%")
        except Exception:
            pass

    def _apply_named_preset(self, name: str):
        self.preset_var.set(name)
        self._apply_preset()

    def _apply_preset(self):
        preset = PRESETS.get(self.preset_var.get())
        if not preset or self.proc is not None:
            return
        locked = ({"embedding_dim", "hidden_size", "num_layers"}
                  if getattr(self, "_arch_locked", False) else set())
        for key, value in preset.items():
            if key in self.vars and key not in locked:
                self.vars[key].set(str(value))
        if not locked and self.opt_tie.get() and \
                preset["embedding_dim"] != preset["hidden_size"]:
            self.opt_tie.set(False)
        note = (" (brain size kept: it belongs to your saved model)"
                if locked else "")
        self.status_label.configure(
            text=f"Preset '{self.preset_var.get()}' applied{note} — review, "
                 f"then press Start (Ctrl+Enter).")

    def _open_dashboard(self):
        script = Path(__file__).resolve().with_name("plot_results.py")
        cmd = [sys.executable, str(script), "--live",
               "--log", str(self._log_file())]
        try:
            subprocess.Popen(cmd)
        except Exception as exc:
            self._append_log(f"[could not open dashboard: {exc}]\n")

    def _open_chat(self):
        script = Path(__file__).resolve().with_name("ask.py")
        ckpt = resolve_checkpoint_path(
            self.vars["checkpoint"].get().strip() or None)
        try:
            subprocess.Popen([sys.executable, str(script), "--gui",
                              "--checkpoint", str(ckpt)])
        except Exception as exc:
            self._append_log(f"[could not open chat: {exc}]\n")

    # ── data-file list helpers ───────────────────────────────────────
    def _parse_data_paths(self, text: str) -> list[str]:
        """Turn the data entry text into a list of paths.

        1. The whole text is itself an existing path (typed or pasted, with
           spaces or Windows backslashes) -> exactly that one path.
        2. Tcl ``splitlist`` (handles ``{braced paths}`` / ``"quoted"``).
        3. ``shlex`` for hand-typed text, then a naive split.
        """
        text = (text or "").strip()
        if not text:
            return []
        bare = text.strip("\"'")
        if bare and os.path.exists(os.path.expanduser(bare)):
            return [os.path.expanduser(bare)]
        try:
            parsed = [str(p) for p in self.root.tk.splitlist(text) if p]
            if parsed:
                return parsed
        except Exception:
            pass
        try:
            import shlex
            return [p for p in shlex.split(text) if p]
        except ValueError:
            return [p for p in text.split() if p]

    def _set_data_paths(self, paths: list[str]) -> None:
        """Store ``paths`` in the data entry using Tcl-list quoting.

        NOTE: never pass ``tk.call("list", ...)`` straight to a StringVar —
        Tkinter returns a *tuple* for list results, and ``StringVar.set``
        would store its repr, e.g. ``('/some/folder',)``. ``tk.merge`` builds
        a proper, correctly-quoted Tcl list *string*.
        """
        paths = [str(p) for p in paths if p]
        if not paths:
            self.data_var.set("")
            return
        try:
            text = self.root.tk.merge(*paths)
            if not isinstance(text, str):
                raise TypeError("tk.merge did not return a string")
        except Exception:
            text = " ".join(
                "{" + p + "}" if any(c.isspace() for c in p) else p
                for p in paths)
        self.data_var.set(text)

    # ── callbacks ────────────────────────────────────────────────────
    def _browse_data(self):
        from tkinter import filedialog, messagebox
        if self.proc is not None:
            messagebox.showinfo(
                "Training in progress",
                "Wait for the current run to finish before changing data "
                "files.",
                parent=self.root)
            return
        try:
            files = filedialog.askopenfilenames(
                parent=self.root,
                title="Add training files (text, JSON, PDF)",
                filetypes=[("Training data",
                            "*.txt *.md *.text *.json *.jsonl *.pdf"),
                           ("Text files", "*.txt *.md *.text"),
                           ("JSON", "*.json *.jsonl"),
                           ("PDF", "*.pdf"),
                           ("All files", "*.*")])
        except Exception as exc:
            messagebox.showerror(
                "File dialog failed", str(exc), parent=self.root)
            return
        if not files:
            return
        if isinstance(files, str):
            try:
                files = tuple(self.root.tk.splitlist(files))
            except Exception:
                files = (files,)
        existing = self._parse_data_paths(self.data_var.get())
        added = 0
        for f in files:
            if f and f not in existing:
                existing.append(f)
                added += 1
        if added:
            self._set_data_paths(existing)
            self.status_label.configure(
                text=f"Data: added {added} file(s) — {len(existing)} total")
        else:
            self.status_label.configure(
                text="Data: selection already in list (nothing to add)")

    def _browse_dir(self):
        from tkinter import filedialog, messagebox
        if self.proc is not None:
            messagebox.showinfo(
                "Training in progress",
                "Wait for the current run to finish before changing data.",
                parent=self.root)
            return
        try:
            folder = filedialog.askdirectory(
                parent=self.root, mustexist=True,
                title="Add a folder of training text files")
        except Exception as exc:
            messagebox.showerror(
                "Folder dialog failed", str(exc), parent=self.root)
            return
        if not folder:
            return
        existing = self._parse_data_paths(self.data_var.get())
        if folder in existing:
            self.status_label.configure(text="Data: folder already in list")
            return
        existing.append(folder)
        self._set_data_paths(existing)
        self.status_label.configure(
            text=f"Data: added folder — {len(existing)} spec(s) total")

    def _clear_data(self):
        if self.proc is not None:
            return
        self.data_var.set("")
        self.status_label.configure(text="Data: file list cleared")

    def _validate(self) -> bool:
        from tkinter import messagebox
        int_keys = ["steps", "batch_size", "block_size", "num_layers",
                    "embedding_dim", "hidden_size", "warmup_steps",
                    "eval_interval", "eval_batches", "grad_accum",
                    "sample_every", "patience", "num_threads"]
        float_keys = ["dropout", "learning_rate", "min_lr_ratio",
                      "weight_decay", "label_smoothing", "val_fraction",
                      "ema_decay", "grad_clip", "resume_lr_scale"]
        try:
            for k in int_keys:
                v = int(self.vars[k].get())
                if k in {"steps", "batch_size", "num_layers", "grad_accum",
                         "embedding_dim", "hidden_size", "eval_interval",
                         "eval_batches"} and v < 1:
                    raise ValueError(f"{k} must be >= 1")
                if k == "block_size" and v < 8:
                    raise ValueError("block_size must be >= 8")
                if k in {"num_threads", "sample_every", "patience",
                         "warmup_steps"} and v < 0:
                    raise ValueError(f"{k} must be >= 0")
            for k in float_keys:
                v = float(self.vars[k].get())
                if k == "dropout" and not (0.0 <= v < 1.0):
                    raise ValueError("dropout must be in [0, 1)")
                if k == "learning_rate" and v <= 0:
                    raise ValueError("learning_rate must be > 0")
                if k == "min_lr_ratio" and not (0 <= v <= 1):
                    raise ValueError("min_lr_ratio must be in [0, 1]")
                if k == "weight_decay" and v < 0:
                    raise ValueError("weight_decay must be >= 0")
                if k == "label_smoothing" and v < 0:
                    raise ValueError("label_smoothing must be >= 0")
                if k == "ema_decay" and not (0.0 <= v < 1.0):
                    raise ValueError("ema_decay must be in [0, 1) "
                                     "(0 disables)")
                if k == "grad_clip" and v < 0:
                    raise ValueError("grad_clip must be >= 0 (0 disables)")
                if k == "resume_lr_scale" and v <= 0:
                    raise ValueError("resume_lr_scale must be > 0")
                if k == "val_fraction" and not (0.0 < v < 1.0):
                    raise ValueError("val_fraction must be in (0, 1)")
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith(("invalid literal", "could not convert")):
                msg = f"Please enter a valid number for '{k}'."
            messagebox.showerror("Invalid hyperparameter", msg)
            return False
        data_paths = self._parse_data_paths(self.data_var.get())
        if data_paths:
            n_files = len(expand_data_specs(
                data_paths, _parse_extensions(DEFAULT_DATA_EXTS_STR),
                verbose=False))
            if n_files == 0:
                messagebox.showerror(
                    "No training data",
                    "None of the selected files or folders contain "
                    f"{', '.join(DEFAULT_DATA_EXTS)} files:\n\n"
                    + "\n".join(data_paths[:8]))
                return False
            self.status_label.configure(
                text=f"Data: {n_files:,} file(s) found")
        if self.opt_tie.get() and int(self.vars["embedding_dim"].get()) != \
                int(self.vars["hidden_size"].get()):
            messagebox.showerror(
                "Invalid tie",
                "Tie weights requires embedding_dim == hidden_size.")
            return False
        return True

    def _build_cmd(self) -> tuple[list[str], Path | None]:
        """Assemble the training subprocess command line.

        Returns ``(cmd, temp_manifest)`` — the caller must eventually pass
        ``temp_manifest`` to :meth:`_cleanup_temp_manifest`.
        """
        # ``--cli`` is REQUIRED here: the default entry point is the GUI,
        # so without it the spawned training subprocess would pop another
        # GUI instead of training headlessly.
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--cli"]
        data_paths = self._parse_data_paths(self.data_var.get())
        tmp_manifest: Path | None = None

        # Windows' command line caps at 32 KB; Linux around 2 MB. When the
        # spec list is large, hand off via a temp manifest so we never hit
        # ARG_MAX — the loader reads it exactly as if it were --data.
        if len(data_paths) > 200:
            fd, tmpname = tempfile.mkstemp(prefix="tiny_text_data_",
                                           suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write("# Auto-generated by the train.py GUI.\n")
                for p in data_paths:
                    f.write(p + "\n")
            tmp_manifest = Path(tmpname)
            cmd += ["--data-list", str(tmp_manifest)]
        elif data_paths:
            cmd += ["--data", *data_paths]

        for key in ["checkpoint", "log", "embedding_dim", "hidden_size",
                    "num_layers", "dropout", "steps", "batch_size",
                    "block_size", "learning_rate", "min_lr_ratio",
                    "warmup_steps", "weight_decay", "label_smoothing",
                    "val_fraction", "eval_interval", "eval_batches",
                    "grad_accum", "sample_every", "patience",
                    "num_threads", "ema_decay", "grad_clip",
                    "resume_lr_scale"]:
            cmd += [f"--{key.replace('_', '-')}", self.vars[key].get()]
        self.run_id = uuid.uuid4().hex[:8]
        cmd += ["--run-id", self.run_id]
        if self.vars["json_keys"].get().strip():
            cmd += ["--json-keys", self.vars["json_keys"].get().strip()]
        cmd += ["--device", _device_value_from_display(self.device_var.get())]
        if self.opt_tie.get():
            cmd.append("--tie-weights")
        if not self.opt_unigram.get():
            cmd.append("--no-unigram-bias-init")
        if not self.opt_live.get():
            cmd.append("--no-live")
        if not self.opt_auto_chat.get():
            cmd.append("--no-auto-chat")
        if not self.opt_resume.get():
            cmd.append("--fresh")
        self._stop_file = Path(tempfile.gettempdir()) / \
            f"ai_trainer_stop_{self.run_id}.flag"
        cmd += ["--stop-file", str(self._stop_file)]
        return cmd, tmp_manifest

    def _start(self):
        if self.proc is not None or not self._validate():
            return
        self.log_text.configure(state=self.tk.NORMAL)
        self.log_text.delete("1.0", self.tk.END)
        self.log_text.configure(state=self.tk.DISABLED)
        self.total_steps = int(self.vars["steps"].get())
        self.progress["value"] = 0
        self._reset_cards()
        self.status_label.configure(text="Starting…")
        self.start_btn.configure(state=self.tk.DISABLED)
        self.stop_btn.configure(state=self.tk.NORMAL)
        self.resume_cb.configure(state=self.tk.DISABLED)
        self.train_start_ts = time.time()

        cmd, self._temp_manifest = self._build_cmd()
        self._append_log(f"$ {_shell_join(cmd)}\n\n")
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=str(Path(__file__).resolve().parent))
        except Exception as exc:
            self._append_log(f"[failed to start: {exc}]\n")
            self._cleanup_temp_manifest()
            self._reset_ui()
            return
        threading.Thread(target=self._reader, daemon=True).start()
        self.root.after(120, self._poll_log)

    def _reader(self):
        try:
            assert self.proc is not None
            for line in self.proc.stdout:
                self.log_queue.put(line.rstrip("\n"))
        finally:
            self.log_queue.put(None)

    def _poll_log(self):
        finished = False
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line is None:
                    finished = True
                    break
                self._append_log(line + "\n")
                self._update_progress(line)
                self._update_metrics(line)
        except queue.Empty:
            pass
        self._poll_csv()
        if finished:
            rc = self.proc.wait() if self.proc else 0
            self._append_log(f"\n[process exited with code {rc}]\n")
            if rc == 0:
                self.progress["value"] = 100.0
                self.status_label.configure(text="Training complete ✓")
            else:
                self.status_label.configure(text=f"Exited with code {rc}")
            self._reset_ui()
            self._cleanup_temp_manifest()
            return
        self.root.after(120, self._poll_log)

    def _update_progress(self, line: str):
        sm = _SESSION_LINE.match(line)
        if sm:
            self.step_offset = int(sm.group(1))
            self.cards["session"].configure(text=f"#{sm.group(2)}")
            self.cards["total"].configure(text=f"{self.step_offset:,}")
            return
        m = re.search(r"step\s+(\d+)\s+\|", line)
        if m and self.total_steps > 0:
            self._set_progress(int(m.group(1)))

    @staticmethod
    def _log_tag(text: str) -> str | None:
        low = text.lstrip().lower()
        if text.startswith("$"):
            return "cmd"
        if "traceback" in low or low.startswith("error") or "[failed" in low:
            return "err"
        if low.startswith("warning") or "(could not" in low:
            return "warn"
        if ("saved best checkpoint" in low or low.startswith("training finished")
                or "saved final" in low or "training complete" in low):
            return "ok"
        if low.startswith("step"):
            return "metric"
        return None

    def _append_log(self, text: str):
        self.log_text.configure(state=self.tk.NORMAL)
        self.log_text.insert(self.tk.END, text, self._log_tag(text) or ())
        # Keep the widget responsive on very long runs.
        if int(self.log_text.index("end-1c").split(".")[0]) > 5000:
            self.log_text.delete("1.0", "1000.0")
        self.log_text.see(self.tk.END)
        self.log_text.configure(state=self.tk.DISABLED)

    def _stop(self, silent: bool = False):
        from tkinter import messagebox
        if self.proc is None:
            return
        if not silent:
            if not messagebox.askyesno(
                    "Stop training?",
                    "Stop the training subprocess?\nThe best checkpoint so "
                    "far is preserved."):
                return
        # POSIX: SIGINT is what Ctrl+C sends — train.py catches it as
        # KeyboardInterrupt, runs its finally block and exits cleanly.
        # Windows has no equivalent signal, so train.py watches for a "stop
        # file" instead (checked every step); terminate() is the last resort.
        try:
            if sys.platform == "win32":
                if self._stop_file is not None:
                    self._stop_file.write_text("stop")
                self.status_label.configure(
                    text="Stopping after the current step…")
                self.root.after(60000, self._force_kill)
            else:
                self.proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    def _force_kill(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _cleanup_temp_manifest(self):
        if self._temp_manifest is not None:
            try:
                self._temp_manifest.unlink(missing_ok=True)
            except OSError:
                pass
            self._temp_manifest = None

    def _on_close(self):
        """Close the window — stop training first if it's still running."""
        from tkinter import messagebox
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno(
                    "Training in progress",
                    "Training is still running. Stop it and close?\n"
                    "The best checkpoint so far is preserved."):
                return
            self._stop(silent=True)
        self._cleanup_temp_manifest()
        self.root.destroy()

    def _show_help(self):
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title("Guide — plain-English reference")
        win.configure(bg=GUI_BG)
        win.geometry("760x680")
        text = tk.Text(win, bg=GUI_PANEL, fg=GUI_TEXT, font=(GUI_SANS[0], 10),
                       wrap=tk.WORD, padx=18, pady=14, relief=tk.FLAT)
        text.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        text.tag_configure("h", foreground=GUI_ACCENT,
                           font=(GUI_SANS[0], 12, "bold"), spacing1=10,
                           spacing3=4)
        text.tag_configure("k", foreground=GUI_ACCENT,
                           font=(GUI_SANS[0], 10, "bold"), spacing1=8)

        def h(t):
            text.insert(tk.END, t + "\n", "h")

        h("QUICK START")
        text.insert(tk.END,
                    "1.  Put your text files in the 'data_to_train_on' folder "
                    "(or use Add files / Add folder).\n"
                    "2.  Click  ★ Use recommended.\n"
                    "3.  Press  ▶ Start training.\n"
                    "4.  Watch VAL LOSS. Lower is smarter. When it stops "
                    "improving, you are done — or train again: every new "
                    "session CONTINUES from where the last one ended, so the "
                    "AI keeps getting smarter.\n")
        h("DOES TRAINING ACCUMULATE?")
        text.insert(tk.END,
                    "Yes. With 'Continue training the existing model' ticked "
                    "(the default), each session starts from the saved AI, "
                    "not from zero. TOTAL STEPS keeps growing. The saved AI "
                    "is only replaced when the new version scores better on "
                    "the practice exam, so a bad session can never make it "
                    "worse. Untick the box to start from zero — the old AI "
                    "is backed up first.\n")
        h("HOW TO READ THE NUMBERS")
        text.insert(tk.END,
                    "• VAL LOSS — mistakes on text the AI has never studied. "
                    "The one that matters. Lower = smarter.\n"
                    "• TRAIN LOSS — mistakes on text it is studying. Should "
                    "fall.\n"
                    "• OVERFIT GAP — if it keeps growing while VAL LOSS "
                    "stops improving, the AI is memorising: add more text.\n"
                    "• PERPLEXITY / BITS PER CHAR — the same score in "
                    "friendlier units.\n")
        h("EVERY SETTING")
        for key, help_text in HELP.items():
            if key.startswith("card_"):
                continue
            text.insert(tk.END, HELP_TITLE.get(key, key) + "\n", "k")
            text.insert(tk.END, help_text + "\n")
        h("HOW TO MAKE THE AI SMARTER (in order of impact)")
        text.insert(tk.END,
                    "1. MORE (and cleaner) TEXT — nothing beats this.\n"
                    "2. KEEP TRAINING — run more sessions; they add up.\n"
                    "3. BIGGER 'Block size' (e.g. 512) so it remembers "
                    "further back; halve 'Batch size' to save memory.\n"
                    "4. If VAL LOSS rises while TRAIN LOSS falls: more "
                    "text, higher Dropout, or Label smoothing 0.05.\n"
                    "5. If the loss graph spikes: halve the Learning rate.\n"
                    "6. Leave 'Smoothed weights (EMA)' and 'Smart start' on.\n")
        text.configure(state=tk.DISABLED)
        tk.Button(win, text="Close", command=win.destroy,
                  bg=GUI_PANEL2, fg=GUI_TEXT, relief=tk.FLAT,
                  padx=20, pady=6).pack(pady=(0, 12))

    def _reset_ui(self):
        self.proc = None
        self.train_start_ts = None
        try:
            self.root.title("Ai-Trainer---chatter")
        except Exception:
            pass
        self.start_btn.configure(state=self.tk.NORMAL)
        self.stop_btn.configure(state=self.tk.DISABLED)
        try:
            self.resume_cb.configure(state=self.tk.NORMAL)
            self._refresh_model_info()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────
#  Argument parser & entry point
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Control
    p.add_argument("--gui", action="store_true",
                   help="Open the training launcher GUI (this is the default; "
                        "kept for backwards compatibility).")
    p.add_argument("--cli", action="store_true",
                   help="Force headless CLI training instead of opening the "
                        "GUI.")
    p.add_argument("--auto-chat", action="store_true", default=True,
                   help="Open the chat window when training finishes "
                        "(default)")
    p.add_argument("--no-auto-chat", dest="auto_chat", action="store_false")
    p.add_argument("--no-live", action="store_true",
                   help="Do not open the live dashboard window")
    # Device
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda:N  (default: auto)")
    p.add_argument("--num-threads", type=int, default=0,
                   help="CPU worker threads. 0 = let PyTorch choose")
    # Data & paths
    p.add_argument("--data", nargs="*", default=None,
                   help="Files, directories, or glob patterns. Directories "
                        "and glob matches are walked recursively for "
                        "--data-ext files. Use --data-list for corpora that "
                        "exceed the shell's argument limit.")
    p.add_argument("--data-list", action="append", default=None, metavar="FILE",
                   help="Text file listing one data path per line "
                        "(# comments and blank lines ignored). Relative paths "
                        "resolve against the manifest's directory. "
                        "Repeatable — use this for corpora of 10k+ files.")
    p.add_argument("--json-keys", default="",
                   help="Comma-separated JSON keys to read, e.g. "
                        "'text,content'. Default: every string value in "
                        ".json/.jsonl files.")
    p.add_argument("--data-ext", default=",".join(DEFAULT_DATA_EXTS),
                   help="Comma-separated extensions picked up when expanding "
                        "directories or globs (default: .txt,.text,.md). "
                        "Pass an empty string to accept any file.")
    p.add_argument("--data-encoding", default="utf-8",
                   help="Text encoding for reading data files "
                        "(default: utf-8).")
    p.add_argument("--data-errors", default="replace",
                   choices=("strict", "replace", "ignore"),
                   help="Handling of undecodable bytes (default: replace — "
                        "bad bytes become U+FFFD; 'strict' aborts the load).")
    p.add_argument("--data-shuffle", action="store_true",
                   help="Shuffle the file order (seeded by --seed). "
                        "Default is sorted, for deterministic runs.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_NAME,
                   help="Where to save the best model. Relative paths are "
                        "anchored to this script's folder, not the "
                        "current directory.")
    p.add_argument("--log", default="training_log.csv")
    p.add_argument("--run-id", default=str(uuid.uuid4())[:8])
    p.add_argument("--seed", type=int, default=42)
    # Schedule
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 = no clipping). Default 1.0")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Exponential moving average of the weights; the "
                        "averaged model is what gets evaluated and saved. "
                        "0 disables. Default 0.999")
    # Architecture
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie-weights", action="store_true")
    p.add_argument("--unigram-bias-init", action="store_true", default=True)
    p.add_argument("--no-unigram-bias-init", dest="unigram_bias_init",
                   action="store_false")
    # Validation
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--patience", type=int, default=0)
    # Continuing a saved model
    p.add_argument("--resume", choices=("auto", "yes", "no"), default="auto",
                   help="auto (default): continue the saved model if there is "
                        "one. yes: require one. no: start from zero (the old "
                        "file is backed up first).")
    p.add_argument("--fresh", action="store_true",
                   help="Shortcut for --resume no: ignore the saved model and "
                        "start from zero (it is backed up first).")
    p.add_argument("--resume-lr-scale", type=float, default=0.5,
                   help="Multiply the peak learning rate by this when "
                        "continuing an already-trained model (default 0.5).")
    p.add_argument("--stop-file", default=None, help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_parser().parse_args()

    # --cli is an explicit request for headless training; it wins even if
    # --gui was also passed.
    if args.cli:
        return run_training(args)

    # Default path: open the GUI (pre-filled with any CLI args the user
    # supplied). Fall back to headless training if there's no display.
    if _can_open_gui():
        launch_training_gui(defaults=vars(args))
        return 0

    print("(No display available — falling back to CLI training.)\n")
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())