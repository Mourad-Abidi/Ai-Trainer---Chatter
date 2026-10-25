"""Chat with a trained CharGRU checkpoint.

Usage
-----
    python ask.py                            # streaming chat window (GUI)
    python ask.py --cli                      # terminal REPL
    python ask.py --cli --qa                 # terminal, Q&A mode on
    python ask.py --checkpoint foo.pt        # custom checkpoint
    (the checkpoint is found automatically whichever folder you run this from;
     if none exists yet you get a prompt to train one or browse for one)
    python ask.py --no-settings              # ignore ~/.ai_trainer---chatter.json
    python ask.py --gui                      # accepted for train.py compat (no-op)

The GUI is a proper streaming chat window: token-by-token generation,
stop button, regenerate, edit-last, adjustable sampling, seed control,
transcript export, hover tooltips everywhere, persistent settings.

The CLI is a full-featured REPL with slash commands — type /help once
you're inside.

Falls back to CLI automatically when no display is available.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

from model import (
    CharGRU,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DATA_DIRNAME,
    PROJECT_DIR,
    find_checkpoint,
    resolve_checkpoint_path,
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except ImportError:
    tk = None
    TK_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────
#  Settings persistence
# ─────────────────────────────────────────────────────────────────────────
SETTINGS_PATH = Path.home() / ".ai_trainer---chatter.json"
DEFAULTS = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "repeat_penalty": 1.0,
    "max_tokens": 200,
    "seed": "",
    "qa_mode": False,
    "stop_at_newline": False,
    "font_size": 11,
    "wrap": True,
}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────
#  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────
def _fmt_secs(s: float) -> str:
    if not math.isfinite(s) or s <= 0:
        return "--"
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def _can_open_gui() -> bool:
    if not TK_AVAILABLE or tk is None:
        return False
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


def _resolve_device_string(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        try:
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        except Exception:
            return f"cuda:{idx}"
    return device.type


# ─────────────────────────────────────────────────────────────────────────
#  CLI REPL
# ─────────────────────────────────────────────────────────────────────────
class Colors:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.enabled else s

    def bold(self, s):     return self._w("1", s)
    def dim(self, s):      return self._w("2", s)
    def italic(self, s):   return self._w("3", s)
    def red(self, s):      return self._w("31", s)
    def green(self, s):    return self._w("1;32", s)
    def yellow(self, s):   return self._w("1;33", s)
    def cyan(self, s):     return self._w("1;36", s)
    def magenta(self, s):  return self._w("1;35", s)
    def purple(self, s):   return self._w("38;5;141", s)


def run_cli(checkpoint_path: Path, args, settings: dict) -> int:
    try:
        model, ckpt, device = CharGRU.from_checkpoint(checkpoint_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    characters = ckpt["characters"]
    stoi = {c: i for i, c in enumerate(characters)}
    c = Colors(sys.stdout.isatty())
    streaming = not args.no_stream

    # CLI-local sampling state, seeded from args then settings
    state = {
        "temperature": args.temperature if args.temperature is not None
                       else settings.get("temperature", 0.7),
        "top_k":       args.top_k if args.top_k is not None
                       else settings.get("top_k", 40),
        "top_p":       args.top_p if args.top_p is not None
                       else settings.get("top_p", 0.95),
        "repeat":      args.repeat_penalty if args.repeat_penalty is not None
                       else settings.get("repeat_penalty", 1.0),
        "max_tokens":  args.max_new_tokens if args.max_new_tokens is not None
                       else settings.get("max_tokens", 200),
        "qa":          args.qa or settings.get("qa_mode", False),
        "stop_nl":     settings.get("stop_at_newline", False),
        "seed":        settings.get("seed", ""),
    }
    history: list[dict] = []

    def _stats_line():
        print(c.dim(
            f"  device={_resolve_device_string(device)}  "
            f"temp={state['temperature']}  top_k={state['top_k'] or 'off'}  "
            f"top_p={state['top_p'] if 0 < state['top_p'] < 1 else 'off'}  "
            f"repeat={state['repeat'] if state['repeat'] != 1 else 'off'}  "
            f"max={state['max_tokens']}  qa={'on' if state['qa'] else 'off'}  "
            f"seed={state['seed'] or 'random'}"))

    print(c.bold("◆  Ai-Trainer---Chatter") + c.dim(" — terminal REPL"))
    print(c.dim(f"  checkpoint: {checkpoint_path.resolve()}"))
    print(c.dim(f"  best val loss: {ckpt.get('best_validation_loss', float('nan')):.4f}"
                f"  |  step {ckpt.get('step', '?')}"
                f"  |  {ckpt.get('sessions', 1)} training session(s)"))
    _stats_line()
    print(c.dim("  type /help for commands, /quit to exit\n"))

    def _apply_seed():
        s = state["seed"].strip()
        if s:
            try:
                v = int(s)
                torch.manual_seed(v)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(v)
            except ValueError:
                pass

    def _save_transcript(path: Path) -> None:
        try:
            ext = path.suffix.lower()
            if ext == ".json":
                path.write_text(json.dumps(
                    {"checkpoint": str(checkpoint_path), "history": history},
                    indent=2), encoding="utf-8")
            else:
                lines = [f"# Ai-Trainer---Chatter/ — {datetime.now():%Y-%m-%d %H:%M}"]
                for h in history:
                    who = "You" if h["role"] == "user" else "AI"
                    lines.append(f"## {who}\n\n{h['content']}")
                path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
            print(c.dim(f"  saved to {path}"))
        except OSError as exc:
            print(c.red(f"  save failed: {exc}"))

    def _handle_command(cmd: str) -> bool:
        """Returns False to quit."""
        parts = cmd.strip().split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        val = parts[1].strip() if len(parts) > 1 else ""

        if name in ("quit", "exit", "q"):
            return False
        if name in ("help", "?", "h"):
            print(c.cyan("  Commands"))
            print("    /help              this message")
            print("    /quit              exit")
            print("    /temp <f>          set temperature (0 = greedy)")
            print("    /topk <n>          set top-k (0 = no filtering)")
            print("    /topp <f>          nucleus sampling, 0-1 (1 = off)")
            print("    /repeat <f>        repetition penalty, 1.0 = off, "
                  "1.1-1.3 curbs loops")
            print("    /max <n>           set max new tokens")
            print("    /qa on|off         Q&A mode")
            print("    /stop_nl on|off    stop generation at newline")
            print("    /seed <n|clear>    set/clear sampling seed")
            print("    /save <path>       save transcript (.md or .json)")
            print("    /clear             clear transcript")
            print("    /stats             show current sampling settings")
            print("    /model             show model info")
            print(c.dim("  Anything else is treated as a prompt."))
            return True
        if name == "stats":
            _stats_line()
            return True
        if name == "model":
            print(c.cyan(f"  checkpoint: {checkpoint_path.resolve()}"))
            try:
                size = checkpoint_path.stat().st_size
                mtime = datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
                print(c.dim(f"  size: {size/1024/1024:.2f} MiB, "
                            f"modified {mtime:%Y-%m-%d %H:%M}"))
            except OSError:
                pass
            print(c.dim(f"  parameters: {_fmt_count(model.num_parameters)}"))
            print(c.dim(f"  vocab: {len(characters)}   device: {_resolve_device_string(device)}"))
            print(c.dim(f"  step {ckpt.get('step', '?')}  "
                        f"best val {ckpt.get('best_validation_loss', float('nan')):.4f}"))
            return True
        if name == "temp":
            try:
                state["temperature"] = max(0.0, float(val))
                print(c.dim(f"  temperature = {state['temperature']}"))
            except ValueError:
                print(c.red("  usage: /temp 0.8"))
            return True
        if name == "topk":
            try:
                state["top_k"] = max(0, int(val))
                print(c.dim(f"  top_k = {state['top_k'] or 'off'}"))
            except ValueError:
                print(c.red("  usage: /topk 40"))
            return True
        if name == "topp":
            try:
                state["top_p"] = min(1.0, max(0.0, float(val)))
                print(c.dim(f"  top_p = {state['top_p']}"))
            except ValueError:
                print(c.red("  usage: /topp 0.95"))
            return True
        if name == "repeat":
            try:
                state["repeat"] = max(1.0, float(val))
                print(c.dim(f"  repetition penalty = {state['repeat']}"))
            except ValueError:
                print(c.red("  usage: /repeat 1.15"))
            return True
        if name == "max":
            try:
                state["max_tokens"] = max(1, int(val))
                print(c.dim(f"  max_tokens = {state['max_tokens']}"))
            except ValueError:
                print(c.red("  usage: /max 200"))
            return True
        if name in ("qa", "stop_nl"):
            key = "qa" if name == "qa" else "stop_nl"
            state[key] = val.lower() in ("on", "1", "true", "yes", "y")
            print(c.dim(f"  {name} = {'on' if state[key] else 'off'}"))
            return True
        if name == "seed":
            if val.lower() in ("clear", "", "none", "random"):
                state["seed"] = ""
                print(c.dim("  seed = random"))
            else:
                try:
                    int(val)
                    state["seed"] = val
                    print(c.dim(f"  seed = {val}"))
                except ValueError:
                    print(c.red("  usage: /seed 42  (or /seed clear)"))
            return True
        if name == "save":
            if not val:
                print(c.red("  usage: /save transcript.md  or  /save t.json"))
            else:
                _save_transcript(Path(val))
            return True
        if name == "clear":
            history.clear()
            print(c.dim("  transcript cleared"))
            return True
        print(c.red(f"  unknown command: /{name}"))
        return True

    while True:
        try:
            raw = input(c.cyan("text> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = raw.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(line[1:]):
                break
            continue

        unknown = sorted(set(line) - set(stoi))
        if unknown:
            print(c.dim(f"  (ignoring characters not in the vocabulary: {unknown})"))
            line = "".join(ch for ch in line if ch in stoi)
            if not line:
                continue

        prompt = f"Q: {line}\nA:" if state["qa"] else line
        prompt_ids = torch.tensor(
            [[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
        stop_id = stoi.get("\n") if (state["qa"] or state["stop_nl"]) else None
        max_new = 64 if state["qa"] else state["max_tokens"]

        _apply_seed()

        print(c.green("model> ") +
              ("" if state["qa"] else c.dim(prompt)), end="", flush=True)
        out_chars: list[str] = []
        t0 = time.time()
        try:
            if streaming:
                for cid in model.generate_tokens(
                        prompt_ids, max_new_tokens=max_new,
                        stop_token_id=stop_id,
                        temperature=state["temperature"],
                        top_k=state["top_k"] or None,
                        top_p=state["top_p"] or None,
                        repetition_penalty=state["repeat"]):
                    ch = characters[cid] if 0 <= cid < len(characters) else "?"
                    out_chars.append(ch)
                    sys.stdout.write(ch)
                    sys.stdout.flush()
            else:
                out = model.generate(
                    prompt_ids, max_new_tokens=max_new,
                    stop_token_id=stop_id,
                    temperature=state["temperature"],
                    top_k=state["top_k"] or None,
                    top_p=state["top_p"] or None,
                    repetition_penalty=state["repeat"])
                tail = out[0].tolist()[len(prompt):]
                out_chars = [characters[i] for i in tail]
                sys.stdout.write("".join(out_chars))
                sys.stdout.flush()
        except Exception as exc:
            print()
            print(c.red(f"  [generation failed: {exc}]"))
            continue
        elapsed = time.time() - t0
        if not state["qa"] and out_chars and not out_chars[-1].endswith("\n"):
            print()
        elif state["qa"]:
            print()
        n = len(out_chars)
        cps = n / elapsed if elapsed > 0 else 0.0
        print(c.dim(f"  [{n} chars · {elapsed:.2f}s · {cps:.0f} c/s]"))

        history.append({"role": "user", "content": line})
        history.append({"role": "model", "content": "".join(out_chars)})

    return 0


# ─────────────────────────────────────────────────────────────────────────
#  Help dictionary — single source of truth for tooltips/popups/help window
# ─────────────────────────────────────────────────────────────────────────
GUI_BG, GUI_PANEL, GUI_PANEL2 = "#0d0d14", "#16161f", "#1c1c28"
GUI_INPUT, GUI_BORDER = "#1c1c28", "#2a2a3a"
GUI_TEXT, GUI_DIM = "#e8e8f0", "#8a8aa0"
GUI_ACCENT, GUI_ACCENT_FG = "#5eead4", "#0d0d14"
GUI_ERR, GUI_NOTICE = "#f87171", "#a78bfa"
GUI_NAME_USER, GUI_NAME_AI = "#5eead4", "#86efac"

HELP: dict[str, str] = {
    "temperature": "The creativity dial. Low = safe and predictable, high = "
                   "wild and surprising. Good value: 0.7.\n\nTechnically: controls randomness. 0 = greedy (always the most likely "
                   "next char). 0.5-0.9 is the sweet spot for coherent-but-"
                   "creative output. Above ~1.2 tends to produce gibberish.",
    "top_k": "Only let the AI pick from its K favourite next letters, so it "
             "never picks a crazy one. Good value: 40.\n\nTechnically: at each step, sample only from the k most likely next characters. "
             "Reduces bizarre rarities without making the output greedy. "
             "0 disables filtering entirely.",
    "top_p": "Another safety filter: ignore the unlikely long tail of letters. "
             "Good value: 0.95 (1.0 = off).\n\nTechnically, nucleus sampling: sample only from the smallest set of "
             "characters whose combined probability reaches p. Adapts to "
             "how confident the model is (unlike top-k). 0.9-0.95 is a "
             "good default; 1.0 disables.",
    "repeat_penalty": "Stops the AI from repeating itself in loops. Keep it at 1.0 "
                      "(off) for maths / Q&A.\n\nTechnically: discourages characters used in the last ~32 "
                      "outputs. 1.0 = off. 1.1-1.3 curbs loops in prose; "
                      "leave it OFF for arithmetic / Q&A data, where "
                      "answers legitimately repeat digits.",
    "max_tokens": "How long the answer may be, counted in letters. Maximum characters to generate per response. Generation may "
                  "stop earlier at a stop token (newline, in Q&A mode).",
    "seed": "A number that makes the AI repeat exactly the same answer every "
            "time (handy for comparing). Leave empty for variety.\n\nTechnically: controls sampling randomness. Same seed + same prompt + same "
            "settings = byte-identical output. Empty = fresh randomness each "
            "time. Click the dice for a new random seed.",
    "qa_mode": "Wraps your input as 'Q: …\\nA:' and stops at the first newline. "
               "Use this when the model was trained on Q&A data.",
    "stop_at_newline": "Stop generating at the first newline. Q&A mode does "
                       "this automatically; use this toggle for normal mode.",
    "send": "Generate a response.\n\n"
            "Shortcuts: Enter or Ctrl+Enter to send, Shift+Enter for a newline.",
    "stop": "Interrupt the current generation. Partial response is kept.\n\n"
            "Shortcut: Esc.",
    "regen": "Discard the last response and regenerate from the same prompt "
             "using fresh sampling. Identical to the original unless the seed "
             "is pinned.",
    "edit_last": "Pull the last prompt back into the input box and remove the "
                 "last exchange from the transcript so you can edit and resend.",
    "clear": "Clear the transcript. Sampling settings and seed are unchanged.",
    "save": "Save the transcript to a file. Use .md for human-readable markdown "
            "or .json for a re-loadable transcript.",
    "copy": "Copy the entire transcript to the clipboard.",
    "help": "Open the full reference: keyboard shortcuts, sampling guide, and "
            "every control explained.",
    "reset": "Restore the sampling controls to defaults: temperature=0.7, "
             "top-k=40, top-p=0.95, repeat penalty off, max=200, no seed, no Q&A, no stop-at-newline.",
    "model": "Everything about the loaded AI: how big it is and how much training it has had "
             "(total steps across all sessions).\n\nFull model card: checkpoint path, file size, mtime, parameters, "
             "vocab size, device, and training metadata.",
    "dice": "Generate a random seed. Fill in a specific seed for reproducible "
            "output across runs.",
}


# ─────────────────────────────────────────────────────────────────────────
#  GUI: tooltip + help badge
# ─────────────────────────────────────────────────────────────────────────
class Tooltip:
    """Hover-tooltip. Optionally opens a full info popup on click.

    ``clickable=False`` is used on widgets that already respond to a click
    (buttons, checkboxes) so their real command isn't shadowed by a modal.
    """

    def __init__(self, widget, text: str, delay: int = 450, clickable: bool = True):
        self.widget = widget
        self.text = text
        self.delay = delay
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
            messagebox.showinfo("Help", self.text, parent=self.widget.winfo_toplevel())
        except Exception:
            pass

    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tip, text=self.text, justify="left",
            background=GUI_PANEL2, foreground=GUI_TEXT,
            relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), padx=10, pady=6, wraplength=360,
        ).pack()
        self._tip = tip

    def _hide(self):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def _help_badge(parent, key: str):
    label = tk.Label(
        parent, text=" ? ", bg=GUI_PANEL2, fg=GUI_ACCENT,
        cursor="hand2", font=("TkDefaultFont", 8, "bold"), padx=3, pady=0,
    )
    Tooltip(label, HELP.get(key, ""))
    return label


# ─────────────────────────────────────────────────────────────────────────
#  GUI: chat window
# ─────────────────────────────────────────────────────────────────────────
class ChatWindow:
    def __init__(self, root, checkpoint_path: Path, settings: dict, qa_default: bool):
        self.root = root
        self.checkpoint_path = Path(checkpoint_path)
        self.settings = dict(DEFAULTS)
        self.settings.update(settings)

        # Model state (populated asynchronously)
        self.model: CharGRU | None = None
        self.characters: list[str] | None = None
        self.stoi: dict[str, int] | None = None
        self.ckpt: dict | None = None
        self.device: torch.device | None = None

        # Sampling state
        self.temperature = tk.DoubleVar(
            value=float(self.settings.get("temperature", 0.7)))
        self.top_k = tk.IntVar(value=int(self.settings.get("top_k", 40)))
        self.top_p = tk.DoubleVar(
            value=float(self.settings.get("top_p", 0.95)))
        self.repeat_penalty = tk.DoubleVar(
            value=float(self.settings.get("repeat_penalty", 1.0)))
        self.max_tokens = tk.IntVar(
            value=int(self.settings.get("max_tokens", 200)))
        self.seed_var = tk.StringVar(value=str(self.settings.get("seed", "")))
        # Respect --qa even when settings file says otherwise.
        self.qa_mode = tk.BooleanVar(
            value=bool(qa_default or self.settings.get("qa_mode", False)))
        self.stop_at_newline = tk.BooleanVar(
            value=bool(self.settings.get("stop_at_newline", False)))
        self.wrap = tk.BooleanVar(value=bool(self.settings.get("wrap", True)))

        # Runtime
        self._busy = False
        self._stop_flag = False
        self._token_queue: queue.Queue = queue.Queue()
        self._history: list[dict] = []  # {"role", "content", "stats"?}
        self._last_prompt: str | None = None
        self._last_stats: dict | None = None
        self._gen_start_ts: float = 0.0
        self._gen_chars: int = 0
        self._last_status_update: float = 0.0
        self._current_ai_text: list[str] = []

        self._mono = self._pick_mono_font(
            int(self.settings.get("font_size", 11)))
        self._build_ui()
        self._load_model_async()

    # ── font helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _pick_mono_font(size: int):
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families())
        except Exception:
            return ("TkFixedFont", size)
        for name in ("JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo",
                     "DejaVu Sans Mono", "Courier New", "Courier"):
            if name in fams:
                return (name, size)
        return ("TkFixedFont", size)

    # ── window / theme ───────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("Ai-Trainer---Chatter/")
        self.root.configure(bg=GUI_BG)
        # Fit the window to the screen (the sampling sidebar needs ~800 px).
        _h = min(840, max(600, self.root.winfo_screenheight() - 90))
        _w = min(1180, max(900, self.root.winfo_screenwidth() - 40))
        self.root.geometry(f"{_w}x{_h}")
        self.root.minsize(900, min(_h, 680))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _e: self._on_stop())
        self.root.bind("<Control-s>", lambda _e: self._on_save())
        self.root.bind("<Control-S>", lambda _e: self._on_save())
        self.root.bind("<Control-l>", lambda _e: self._on_clear())

        self._build_header()
        self._build_body()
        self._build_status_bar()
        self._build_input_area()

    def _build_header(self):
        header = tk.Frame(self.root, bg=GUI_PANEL, height=54)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="◆  Ai-Trainer---Chatter/", bg=GUI_PANEL, fg=GUI_ACCENT,
                 font=("TkDefaultFont", 12, "bold"),
                 padx=16).pack(side=tk.LEFT, pady=14)

        self.model_badge = tk.Label(
            header, text="loading…", bg=GUI_PANEL, fg=GUI_DIM,
            font=("TkDefaultFont", 10), padx=8)
        self.model_badge.pack(side=tk.LEFT, pady=14)
        Tooltip(self.model_badge, HELP["model"])  # badge stays clickable

        for text, cmd, key in [
            ("?  Help", self._on_help, "help"),
            ("Copy all", self._on_copy_all, "copy"),
            ("Save…", self._on_save, "save"),
            ("Clear", self._on_clear, "clear"),
        ]:
            btn = tk.Button(
                header, text=text, command=cmd,
                bg=GUI_PANEL2, fg=GUI_TEXT, relief=tk.FLAT,
                font=("TkDefaultFont", 10), padx=14, pady=4, cursor="hand2",
            )
            btn.pack(side=tk.RIGHT, padx=(0, 6), pady=10)
            Tooltip(btn, HELP[key], clickable=False)

    def _build_body(self):
        body = tk.Frame(self.root, bg=GUI_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Transcript
        chat_frame = tk.Frame(body, bg=GUI_BG)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                         padx=(12, 6), pady=(8, 6))
        self.text = tk.Text(
            chat_frame, bg=GUI_BG, fg=GUI_TEXT, font=self._mono,
            relief=tk.FLAT, wrap=tk.WORD if self.wrap.get() else tk.NONE,
            padx=18, pady=14, insertbackground=GUI_ACCENT,
            state=tk.DISABLED, spacing1=2, spacing3=2,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(chat_frame, command=self.text.yview,
                              bg=GUI_PANEL, troughcolor=GUI_BG,
                              relief=tk.FLAT, width=12)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=scroll.set)

        self.text.tag_config("user_name", foreground=GUI_NAME_USER,
                             font=(self._mono[0], self._mono[1], "bold"),
                             spacing1=16)
        self.text.tag_config("user_body", foreground=GUI_TEXT,
                             lmargin1=14, lmargin2=14)
        self.text.tag_config("ai_name", foreground=GUI_NAME_AI,
                             font=(self._mono[0], self._mono[1], "bold"),
                             spacing1=16)
        self.text.tag_config("ai_body", foreground=GUI_TEXT,
                             lmargin1=14, lmargin2=14)
        self.text.tag_config("stats", foreground=GUI_DIM,
                             font=(self._mono[0], self._mono[1] - 2),
                             lmargin1=14, lmargin2=14, spacing1=2)
        self.text.tag_config("notice", foreground=GUI_NOTICE,
                             font=(self._mono[0], self._mono[1] - 1, "italic"),
                             lmargin1=14, lmargin2=14, spacing1=8)
        self.text.tag_config("error", foreground=GUI_ERR,
                             lmargin1=14, lmargin2=14)

        # Context menu
        self.context_menu = tk.Menu(
            self.root, tearoff=0, bg=GUI_PANEL, fg=GUI_TEXT,
            activebackground=GUI_ACCENT, activeforeground=GUI_ACCENT_FG)
        self.context_menu.add_command(label="Copy selection",
                                      command=self._on_copy_selection)
        self.context_menu.add_command(label="Copy all", command=self._on_copy_all)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Clear transcript",
                                      command=self._on_clear)
        self.context_menu.add_command(label="Save as…", command=self._on_save)
        self.text.bind("<Button-3>", self._show_context_menu)
        self.text.bind("<Button-2>", self._show_context_menu)
        self.text.bind("<Control-Button-1>", self._show_context_menu)

        # Sidebar
        self._build_sidebar(body)

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=GUI_PANEL, width=270)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=(8, 6))
        side.pack_propagate(False)

        # sampling
        tk.Label(side, text="SAMPLING", bg=GUI_PANEL, fg=GUI_DIM,
                 font=("TkDefaultFont", 8, "bold")).pack(
            anchor=tk.W, padx=16, pady=(16, 6))

        self._add_slider(side, "Temperature", self.temperature,
                         from_=0.0, to=2.0, resolution=0.05, key="temperature")
        self._add_slider(side, "Top-k", self.top_k,
                         from_=0, to=100, resolution=1, key="top_k")
        self._add_slider(side, "Top-p", self.top_p,
                         from_=0.1, to=1.0, resolution=0.01, key="top_p")
        self._add_slider(side, "Repeat penalty", self.repeat_penalty,
                         from_=1.0, to=1.5, resolution=0.01,
                         key="repeat_penalty")
        self._add_spinbox(side, "Max tokens", self.max_tokens, 1, 2000,
                          key="max_tokens")

        # seed
        seed_row = tk.Frame(side, bg=GUI_PANEL)
        seed_row.pack(fill=tk.X, padx=16, pady=(10, 4))
        hdr = tk.Frame(seed_row, bg=GUI_PANEL)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Seed", bg=GUI_PANEL, fg=GUI_TEXT,
                 font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        _help_badge(hdr, "seed").pack(side=tk.RIGHT, padx=(0, 0))
        seed_inner = tk.Frame(seed_row, bg=GUI_PANEL)
        seed_inner.pack(fill=tk.X, pady=(2, 0))
        entry = tk.Entry(
            seed_inner, textvariable=self.seed_var, bg=GUI_INPUT, fg=GUI_TEXT,
            insertbackground=GUI_ACCENT, relief=tk.FLAT, font=("TkDefaultFont", 10))
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        dice = tk.Button(
            seed_inner, text="🎲", command=self._on_random_seed,
            bg=GUI_PANEL2, fg=GUI_ACCENT, relief=tk.FLAT, padx=8, pady=2,
            font=("TkDefaultFont", 10), cursor="hand2")
        dice.pack(side=tk.RIGHT, padx=(4, 0))
        Tooltip(dice, HELP["dice"], clickable=False)

        tk.Frame(side, bg=GUI_BORDER, height=1).pack(fill=tk.X, padx=16, pady=12)

        # mode
        tk.Label(side, text="MODE", bg=GUI_PANEL, fg=GUI_DIM,
                 font=("TkDefaultFont", 8, "bold")).pack(
            anchor=tk.W, padx=16, pady=(0, 6))
        self._add_checkbox(side, "Q&A (Q:…\\nA:)", self.qa_mode, "qa_mode")
        self._add_checkbox(side, "Stop at newline", self.stop_at_newline,
                           "stop_at_newline")

        tk.Frame(side, bg=GUI_BORDER, height=1).pack(fill=tk.X, padx=16, pady=12)

        # model card
        tk.Label(side, text="MODEL", bg=GUI_PANEL, fg=GUI_DIM,
                 font=("TkDefaultFont", 8, "bold")).pack(
            anchor=tk.W, padx=16, pady=(0, 6))
        self.model_info_frame = tk.Frame(side, bg=GUI_PANEL)
        self.model_info_frame.pack(fill=tk.X, padx=16)
        self._info_labels = {}
        for label in ("Device", "Parameters", "Vocab", "Step", "Best val"):
            row = tk.Frame(self.model_info_frame, bg=GUI_PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, bg=GUI_PANEL, fg=GUI_DIM,
                     font=("TkDefaultFont", 9)).pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text="—", bg=GUI_PANEL, fg=GUI_TEXT,
                               font=("TkDefaultFont", 9))
            val_lbl.pack(side=tk.RIGHT)
            self._info_labels[label] = val_lbl

        tk.Frame(side, bg=GUI_BORDER, height=1).pack(fill=tk.X, padx=16, pady=12)

        reset_btn = tk.Button(
            side, text="Reset to defaults", command=self._on_reset,
            bg=GUI_PANEL2, fg=GUI_TEXT, relief=tk.FLAT,
            font=("TkDefaultFont", 9), padx=12, pady=5, cursor="hand2")
        reset_btn.pack(fill=tk.X, padx=16)
        Tooltip(reset_btn, HELP["reset"], clickable=False)

    def _add_slider(self, parent, label, var, from_, to, resolution, key):
        row = tk.Frame(parent, bg=GUI_PANEL)
        row.pack(fill=tk.X, padx=16, pady=(0, 2))
        hdr = tk.Frame(row, bg=GUI_PANEL)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=label, bg=GUI_PANEL, fg=GUI_TEXT,
                 font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        tk.Label(hdr, textvariable=var, bg=GUI_PANEL, fg=GUI_ACCENT,
                 font=("TkDefaultFont", 10)).pack(side=tk.RIGHT)
        _help_badge(hdr, key).pack(side=tk.RIGHT, padx=(0, 4))
        tk.Scale(
            row, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL,
            variable=var, bg=GUI_PANEL, fg=GUI_TEXT, troughcolor=GUI_INPUT,
            highlightthickness=0, sliderrelief=tk.FLAT, showvalue=False,
            font=("TkDefaultFont", 8), length=230, borderwidth=0,
        ).pack(fill=tk.X)

    def _add_spinbox(self, parent, label, var, lo, hi, key):
        row = tk.Frame(parent, bg=GUI_PANEL)
        row.pack(fill=tk.X, padx=16, pady=(6, 2))
        hdr = tk.Frame(row, bg=GUI_PANEL)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=label, bg=GUI_PANEL, fg=GUI_TEXT,
                 font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        _help_badge(hdr, key).pack(side=tk.RIGHT)
        tk.Spinbox(
            row, from_=lo, to=hi, textvariable=var,
            bg=GUI_INPUT, fg=GUI_TEXT, insertbackground=GUI_ACCENT,
            relief=tk.FLAT, font=("TkDefaultFont", 10),
            buttonbackground=GUI_PANEL2,
        ).pack(fill=tk.X, pady=(2, 0))

    def _add_checkbox(self, parent, label, var, key):
        row = tk.Frame(parent, bg=GUI_PANEL)
        row.pack(fill=tk.X, padx=16, pady=1)
        tk.Checkbutton(
            row, text=label, variable=var, bg=GUI_PANEL, fg=GUI_TEXT,
            selectcolor=GUI_INPUT, activebackground=GUI_PANEL,
            activeforeground=GUI_TEXT, font=("TkDefaultFont", 10),
            anchor=tk.W, relief=tk.FLAT, highlightthickness=0, bd=0,
        ).pack(side=tk.LEFT)
        _help_badge(row, key).pack(side=tk.RIGHT)

    def _build_input_area(self):
        bottom = tk.Frame(self.root, bg=GUI_PANEL)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # input text
        self.input = tk.Text(
            bottom, bg=GUI_INPUT, fg=GUI_TEXT, insertbackground=GUI_ACCENT,
            relief=tk.FLAT, font=self._mono, height=3, wrap=tk.WORD,
            padx=12, pady=8,
        )
        self.input.pack(fill=tk.X, padx=16, pady=(10, 6))
        self.input.bind("<Return>", self._on_input_return)
        self.input.configure(state=tk.DISABLED)

        # buttons
        btn_row = tk.Frame(bottom, bg=GUI_PANEL)
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.stop_btn = self._make_btn(btn_row, "Stop", self._on_stop,
                                       bg=GUI_PANEL2, fg=GUI_ERR, key="stop")
        self.stop_btn.configure(state=tk.DISABLED)
        self.regen_btn = self._make_btn(btn_row, "Regenerate",
                                        self._on_regenerate,
                                        bg=GUI_PANEL2, fg=GUI_TEXT, key="regen")
        self.regen_btn.configure(state=tk.DISABLED)
        self.edit_btn = self._make_btn(btn_row, "Edit last", self._on_edit_last,
                                       bg=GUI_PANEL2, fg=GUI_TEXT,
                                       key="edit_last")
        self.edit_btn.configure(state=tk.DISABLED)

        self.send_btn = tk.Button(
            btn_row, text="Send  ⏎", command=self._send,
            bg=GUI_ACCENT, fg=GUI_ACCENT_FG, relief=tk.FLAT,
            font=("TkDefaultFont", 10, "bold"), padx=22, pady=6,
            cursor="hand2", state=tk.DISABLED,
        )
        self.send_btn.pack(side=tk.RIGHT)
        Tooltip(self.send_btn, HELP["send"], clickable=False)

    def _make_btn(self, parent, text, cmd, bg, fg, key):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, relief=tk.FLAT,
            font=("TkDefaultFont", 10), padx=14, pady=5, cursor="hand2",
        )
        btn.pack(side=tk.LEFT, padx=(0, 6))
        _help_badge(parent, key).pack(side=tk.LEFT, padx=(0, 14))
        Tooltip(btn, HELP[key], clickable=False)
        return btn

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=GUI_BG, height=26)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        self.status = tk.Label(
            bar, text="", bg=GUI_BG, fg=GUI_DIM,
            font=("TkDefaultFont", 9), anchor=tk.W, padx=16)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_right = tk.Label(
            bar, text="", bg=GUI_BG, fg=GUI_DIM,
            font=("TkDefaultFont", 9), anchor=tk.E, padx=16)
        self.status_right.pack(side=tk.RIGHT)

    # ── model loading ────────────────────────────────────────────────────
    def _load_model_async(self):
        self._set_status("Loading model…")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            model, ckpt, device = CharGRU.from_checkpoint(self.checkpoint_path)
            # Bind values as default args so the callback doesn't depend on
            # local scopes that may be gone by the time Tk runs it.
            self.root.after(
                0, lambda m=model, c=ckpt, d=device:
                self._on_model_ready(m, c, d))
        except Exception as exc:
            # `exc` is deleted at the end of this except-block; capture the
            # message as a plain string before scheduling the callback.
            err = str(exc)
            self.root.after(0, lambda msg=err: self._on_model_error(msg))

    def _on_model_ready(self, model, ckpt, device):
        self.model = model
        self.ckpt = ckpt
        self.device = device
        self.characters = ckpt["characters"]
        self.stoi = {c: i for i, c in enumerate(self.characters)}
        self._update_model_card()
        self.input.configure(state=tk.NORMAL)
        self.input.focus_set()
        self.send_btn.configure(state=tk.NORMAL)
        self._set_status("Ready")
        self._welcome()

    def _on_model_error(self, exc_msg: str):
        self._append_notice(f"Failed to load model: {exc_msg}", error=True)
        self._set_status("Model failed to load")

    def _update_model_card(self):
        if self.ckpt is None or self.model is None:
            return
        step = self.ckpt.get("step", "?")
        bvl = self.ckpt.get("best_validation_loss", float("nan"))
        values = {
            "Device": _resolve_device_string(self.device),
            "Parameters": _fmt_count(self.model.num_parameters),
            "Vocab": str(len(self.characters)),
            "Step": _fmt_count(step) if isinstance(step, int) else str(step),
            "Best val": f"{bvl:.4f}" if isinstance(bvl, float) else str(bvl),
        }
        for k, v in values.items():
            self._info_labels[k].configure(text=v)
        summary = (f"{values['Device']}  ·  {values['Parameters']} params  ·  "
                   f"step {values['Step']}  ·  val {values['Best val']}")
        self.model_badge.configure(text=summary)
        Tooltip(self.model_badge, self._full_model_card())

    def _full_model_card(self) -> str:
        if not self.ckpt:
            return HELP["model"]
        lines = [f"Checkpoint: {self.checkpoint_path.resolve()}"]
        try:
            st = self.checkpoint_path.stat()
            lines.append(f"Size: {st.st_size/1024/1024:.2f} MiB")
            lines.append(f"Modified: {datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")
        except OSError:
            pass
        lines.append(f"Device: {_resolve_device_string(self.device)}")
        lines.append(f"Parameters: {_fmt_count(self.model.num_parameters)}")
        lines.append(f"Vocabulary: {len(self.characters)} characters")
        lines.append(f"Architecture: "
                     f"emb={self.ckpt.get('embedding_dim')}, "
                     f"hidden={self.ckpt.get('hidden_size')}, "
                     f"layers={self.ckpt.get('num_layers')}")
        lines.append(f"Trained: {self.ckpt.get('step', '?')} total steps over "
                     f"{self.ckpt.get('sessions', 1)} session(s), "
                     f"best val loss {self.ckpt.get('best_validation_loss', float('nan')):.4f}")
        return "\n".join(lines)

    # ── transcript helpers ───────────────────────────────────────────────
    def _at_bottom(self) -> bool:
        try:
            return self.text.yview()[1] > 0.98
        except Exception:
            return True

    def _insert(self, text: str, tag: str, scroll: bool = True):
        at_bottom = self._at_bottom()
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, text, tag)
        self.text.configure(state=tk.DISABLED)
        if scroll and at_bottom:
            self.text.see(tk.END)

    def _append_user(self, text: str):
        self._insert("\nYou\n", "user_name")
        self._insert(text + "\n", "user_body")

    def _append_ai_header(self):
        self._insert("\nAI\n", "ai_name")

    def _append_notice(self, text: str, error: bool = False):
        self._insert("\n" + text + "\n", "error" if error else "notice")

    def _append_stats(self, stats: dict):
        self._insert(
            f"  [{stats['n_chars']} chars · {stats['elapsed']:.2f}s · "
            f"{stats['cps']:.0f} c/s]\n", "stats")

    def _remove_last_exchange(self):
        """Remove the last user+model exchange from history *and* transcript.

        History is the source of truth: if there's nothing to remove there,
        we leave the widget alone too.
        """
        had_model = bool(self._history and self._history[-1]["role"] == "model")
        if had_model:
            self._history.pop()
        had_user = bool(self._history and self._history[-1]["role"] == "user")
        if had_user:
            self._history.pop()
        if not (had_model or had_user):
            return
        ranges = self.text.tag_ranges("user_name")
        if len(ranges) >= 2:
            self.text.configure(state=tk.NORMAL)
            self.text.delete(ranges[-2], tk.END)
            self.text.configure(state=tk.DISABLED)

    def _welcome(self):
        step = self.ckpt.get("step", "?")
        bvl = self.ckpt.get("best_validation_loss", float("nan"))
        msg = (
            f"Model loaded — {step} total training steps over "
            f"{self.ckpt.get('sessions', 1)} session(s), best val loss {bvl:.4f}.\n"
            "Type a prompt and press Enter. Esc stops generation. "
            "Hover/click any ? for help."
        )
        self._append_notice(msg)

    # ── send / stream / stop ─────────────────────────────────────────────
    def _on_input_return(self, event):
        if event.state & 0x1:  # Shift
            return None  # allow newline
        self._send()
        return "break"

    def _send(self):
        if self._busy or self.model is None:
            return
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            return

        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            self._append_notice(f"(Ignored characters not in vocabulary: {unknown})")
            text = "".join(c for c in text if c in self.stoi)
            if not text:
                return
        self.input.delete("1.0", tk.END)

        self._history.append({"role": "user", "content": text})
        self._append_user(text)
        self._last_prompt = text
        self._start_generation(text, append_user=False)

    def _on_regenerate(self):
        if self._busy or self.model is None or self._last_prompt is None:
            return
        prompt = self._last_prompt
        self._remove_last_exchange()
        self._history.append({"role": "user", "content": prompt})
        self._start_generation(prompt, append_user=True)

    def _on_edit_last(self):
        if self._busy or self._last_prompt is None:
            return
        self._remove_last_exchange()
        self.input.delete("1.0", tk.END)
        self.input.insert("1.0", self._last_prompt)
        self.input.focus_set()
        self._last_prompt = None
        self.regen_btn.configure(state=tk.DISABLED)
        self.edit_btn.configure(state=tk.DISABLED)

    def _start_generation(self, user_text: str, append_user: bool):
        if append_user:
            self._append_user(user_text)

        # Fresh queue — any stale ("done" or "error") messages left behind by
        # a previous run (e.g. after an exception) must not leak into this one,
        # otherwise _poll would finish this generation on its first tick.
        self._token_queue = queue.Queue()

        prompt = f"Q: {user_text.strip()}\nA:" if self.qa_mode.get() else user_text
        stop_id = None
        if self.qa_mode.get() or self.stop_at_newline.get():
            stop_id = self.stoi.get("\n")

        # Seed
        seed_str = self.seed_var.get().strip()
        if seed_str:
            try:
                seed = int(seed_str)
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            except ValueError:
                self._append_notice(f"(invalid seed '{seed_str}' — using random)")

        prompt_ids = torch.tensor(
            [[self.stoi[c] for c in prompt]],
            dtype=torch.long, device=self.device)

        self._append_ai_header()
        self._current_ai_text = []
        self._busy = True
        self._stop_flag = False
        self._gen_start_ts = time.time()
        self._gen_chars = 0
        self._last_stats = None
        self._set_controls_enabled(False)
        self.stop_btn.configure(state=tk.NORMAL)
        self._set_status("Generating…")

        thread = threading.Thread(
            target=self._generate_worker,
            args=(prompt_ids, int(self.max_tokens.get()),
                  stop_id, float(self.temperature.get()),
                  int(self.top_k.get() or 0),
                  float(self.top_p.get()), float(self.repeat_penalty.get())),
            daemon=True,
        )
        thread.start()
        self.root.after(30, self._poll)

    def _generate_worker(self, prompt_ids, max_new, stop_id, temperature, top_k,
                         top_p=1.0, repeat_penalty=1.0):
        # Count locally so the stats we push don't depend on the main thread
        # having caught up with the queue.
        count = 0
        try:
            for cid in self.model.generate_tokens(
                prompt_ids,
                max_new_tokens=max_new,
                stop_token_id=stop_id,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if 0 < top_p < 1 else None,
                repetition_penalty=repeat_penalty,
                should_stop=lambda: self._stop_flag,
            ):
                if 0 <= cid < len(self.characters):
                    self._token_queue.put(("char", self.characters[cid]))
                    count += 1
        except Exception as exc:
            self._token_queue.put(("error", str(exc)))
        finally:
            elapsed = time.time() - self._gen_start_ts
            stats = {
                "elapsed": elapsed,
                "n_chars": count,
                "cps": (count / elapsed) if elapsed > 0 else 0.0,
            }
            self._token_queue.put(("done", stats))

    def _poll(self):
        chars: list[str] = []
        try:
            while True:
                kind, payload = self._token_queue.get_nowait()
                if kind == "char":
                    chars.append(payload)
                elif kind == "done":
                    # Flush any pending chars before finishing so the final
                    # chunk lands in the widget *and* in `_current_ai_text`
                    # (and therefore in saved transcripts).
                    if chars:
                        chunk = "".join(chars)
                        self._insert_stream(chunk)
                        self._current_ai_text.append(chunk)
                        self._gen_chars += len(chunk)
                    self._on_gen_done(payload)
                    return
                elif kind == "error":
                    # Drain any remaining queue items (notably the trailing
                    # "done" the worker always sends) so they don't leak into
                    # the next generation.
                    self._drain_queue()
                    self._on_gen_error(payload)
                    return
        except queue.Empty:
            pass

        if chars:
            chunk = "".join(chars)
            self._insert_stream(chunk)
            self._current_ai_text.append(chunk)
            self._gen_chars += len(chunk)

        now = time.time()
        if now - self._last_status_update > 0.2:
            self._last_status_update = now
            self._update_live_status()

        if self._busy:
            self.root.after(30, self._poll)

    def _drain_queue(self):
        try:
            while True:
                self._token_queue.get_nowait()
        except queue.Empty:
            pass

    def _insert_stream(self, text: str):
        self._insert(text, "ai_body", scroll=True)

    def _update_live_status(self):
        if not self._busy:
            return
        elapsed = time.time() - self._gen_start_ts
        cps = self._gen_chars / elapsed if elapsed > 0 else 0.0
        self._set_status(f"Generating… {self._gen_chars} chars · "
                         f"{cps:.0f} c/s · {elapsed:.1f}s")

    def _on_gen_done(self, stats: dict):
        self._busy = False
        self._stop_flag = False
        self._last_stats = stats
        self._append_stats(stats)
        self._history.append({
            "role": "model",
            "content": "".join(self._current_ai_text),
            "stats": stats,
        })
        self._current_ai_text = []
        self._set_controls_enabled(True)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status("Ready")
        self.status_right.configure(
            text=f"last: {stats['n_chars']} chars · {stats['elapsed']:.2f}s · "
                 f"{stats['cps']:.0f} c/s")
        self.input.focus_set()

    def _on_gen_error(self, msg: str):
        self._busy = False
        self._stop_flag = False
        self._append_notice(f"[generation error: {msg}]", error=True)
        self._set_controls_enabled(True)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status("Ready (last generation failed)")

    def _on_stop(self):
        if self._busy:
            self._stop_flag = True
            self._set_status("Stopping…")

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for w in (self.send_btn, self.regen_btn, self.edit_btn):
            w.configure(state=state)
        if enabled:
            self.regen_btn.configure(
                state=tk.NORMAL if self._last_prompt else tk.DISABLED)
            self.edit_btn.configure(
                state=tk.NORMAL if self._last_prompt else tk.DISABLED)
        self.input.configure(state=state)

    # ── non-generation buttons ───────────────────────────────────────────
    def _on_clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)
        self._history.clear()
        self._last_prompt = None
        self.regen_btn.configure(state=tk.DISABLED)
        self.edit_btn.configure(state=tk.DISABLED)

    def _on_copy_selection(self):
        try:
            sel = self.text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(sel)
        self._set_status("Copied selection")

    def _on_copy_all(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.text.get("1.0", tk.END))
            self._set_status("Copied to clipboard")
        except tk.TclError:
            pass

    def _on_save(self):
        if not self._history:
            self._set_status("Nothing to save yet")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save conversation",
            defaultextension=".md",
            initialfile=f"chat_{datetime.now():%Y%m%d_%H%M%S}.md",
            filetypes=[("Markdown", "*.md"), ("JSON", "*.json"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        try:
            if p.suffix.lower() == ".json":
                p.write_text(json.dumps({
                    "checkpoint": str(self.checkpoint_path),
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "sampling": {
                        "temperature": self.temperature.get(),
                        "top_k": self.top_k.get(),
                        "top_p": round(self.top_p.get(), 3),
                        "repeat_penalty": round(self.repeat_penalty.get(), 3),
                        "max_tokens": self.max_tokens.get(),
                        "seed": self.seed_var.get(),
                        "qa_mode": self.qa_mode.get(),
                        "stop_at_newline": self.stop_at_newline.get(),
                    },
                    "history": self._history,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                lines = [
                    f"# Ai-Trainer---Chatter/ — {datetime.now():%Y-%m-%d %H:%M}",
                    f"_checkpoint: {self.checkpoint_path.resolve()}_",
                ]
                for h in self._history:
                    who = "You" if h["role"] == "user" else "AI"
                    lines.append(f"## {who}\n\n{h['content']}")
                p.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
            self._set_status(f"Saved to {p.name}")
        except OSError as exc:
            self._append_notice(f"[save failed: {exc}]", error=True)

    def _on_reset(self):
        self.temperature.set(0.7)
        self.top_k.set(40)
        self.top_p.set(0.95)
        self.repeat_penalty.set(1.0)
        self.max_tokens.set(200)
        self.seed_var.set("")
        self.qa_mode.set(False)
        self.stop_at_newline.set(False)
        self._set_status("Sampling reset to defaults")

    def _on_random_seed(self):
        self.seed_var.set(str(random.randint(1, 2**31 - 1)))
        self._set_status("New random seed set")

    def _on_help(self):
        win = tk.Toplevel(self.root)
        win.title("Help — Ai-Trainer---Chatter")
        win.configure(bg=GUI_BG)
        win.geometry("760x680")
        text = tk.Text(
            win, bg=GUI_PANEL, fg=GUI_TEXT, font=("TkFixedFont", 10),
            wrap=tk.WORD, padx=18, pady=14, relief=tk.FLAT,
            insertbackground=GUI_ACCENT,
        )
        text.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        scroll = tk.Scrollbar(win, command=text.yview, bg=GUI_PANEL,
                              troughcolor=GUI_BG, relief=tk.FLAT)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll.set)

        # Sections mix pre-formatted strings (shortcuts / guide) and
        # (key, description) tuples (every control). The renderer below
        # handles both, so we don't crash trying to unpack a string.
        sections = [
            ("Keyboard shortcuts", [
                "Enter              send prompt",
                "Shift+Enter        newline in prompt",
                "Ctrl+Enter         send prompt (alternative)",
                "Esc                stop generation",
                "Ctrl+L             clear transcript",
                "Ctrl+S             save conversation",
                "Right-click        transcript context menu",
            ]),
            ("Sampling guide", [
                "Temperature 0      greedy — always the most likely next char",
                "Temperature 0.7    balanced — creative but coherent (default)",
                "Temperature 1.2+   wilder; often gibberish for small models",
                "Top-k 0            no filtering — sample from the full vocab",
                "Top-k 40           ignore the 40-ish rarest chars each step",
            ]),
            ("Every control", [(k, HELP[k]) for k in HELP]),
        ]
        lines: list[str] = []
        for title, entries in sections:
            lines.append(f"▸ {title}\n" + "─" * (len(title) + 2) + "\n")
            for entry in entries:
                if isinstance(entry, tuple):
                    k, v = entry
                    lines.append(f"  {k}\n  {v}\n")
                else:
                    lines.append(f"  {entry}\n")
            lines.append("")
        text.insert("1.0", "\n".join(lines))
        text.configure(state=tk.DISABLED)
        tk.Button(
            win, text="Close", command=win.destroy,
            bg=GUI_PANEL2, fg=GUI_TEXT, relief=tk.FLAT,
            font=("TkDefaultFont", 10), padx=22, pady=6, cursor="hand2",
        ).pack(pady=(0, 12))

    def _show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _set_status(self, text: str):
        self.status.configure(text=text)

    def _on_close(self):
        self._stop_flag = True
        try:
            save_settings({
                "temperature": float(self.temperature.get()),
                "top_k": int(self.top_k.get()),
                "top_p": round(float(self.top_p.get()), 3),
                "repeat_penalty": round(float(self.repeat_penalty.get()), 3),
                "max_tokens": int(self.max_tokens.get()),
                "seed": self.seed_var.get(),
                "qa_mode": bool(self.qa_mode.get()),
                "stop_at_newline": bool(self.stop_at_newline.get()),
                "font_size": int(self.settings.get("font_size", 11)),
                "wrap": bool(self.wrap.get()),
            })
        except Exception:
            pass
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────
#  GUI launcher
# ─────────────────────────────────────────────────────────────────────────
def launch_gui(checkpoint_path: Path, settings: dict, qa_default: bool) -> None:
    if not TK_AVAILABLE or tk is None:
        print("tkinter is not available. Run with --cli for terminal mode.")
        return
    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"Could not open GUI ({exc}). Run with --cli for terminal mode.")
        return
    ChatWindow(root, checkpoint_path, settings, qa_default)
    root.mainloop()


# ─────────────────────────────────────────────────────────────────────────
#  Parser & entry point
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_NAME,
                   help="Checkpoint to load. Searched in the script folder "
                        "and the current directory.")
    p.add_argument("--cli", action="store_true",
                   help="Force terminal REPL (default is the GUI)")
    p.add_argument("--gui", action="store_true",
                   help="Accepted for compatibility with train.py "
                        "(GUI is already the default).")
    p.add_argument("--no-settings", action="store_true",
                   help="Ignore ~/.ai_trainer---chatter.json")
    p.add_argument("--qa", action="store_true",
                   help="Start with Q&A mode on")
    p.add_argument("--no-stream", action="store_true",
                   help="CLI only: print whole response instead of streaming")

    # CLI sampling defaults (override settings file)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None,
                   help="Nucleus sampling threshold, 0-1 (default 0.95)")
    p.add_argument("--repeat-penalty", type=float, default=None,
                   help="Repetition penalty >= 1 (default 1.0 = off)")
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p


def _missing_checkpoint_text(requested: str) -> str:
    target = resolve_checkpoint_path(requested)
    return (
        "No trained model found.\n"
        f"  looked for : {target}\n"
        f"  also in    : {Path.cwd()}\n\n"
        "Train one first (the chat opens by itself when training ends):\n"
        "    python train.py                                  # launcher GUI\n"
        f"    python train.py --cli --data {DEFAULT_DATA_DIRNAME}"
        "          # headless\n"
        "Or point at an existing file:  python ask.py --checkpoint path/to/model.pt"
    )


def _handle_missing_checkpoint(requested: str, want_gui: bool) -> Path | None:
    """No checkpoint on disk: explain, and (GUI) offer to train or browse.

    Returns a usable checkpoint path, or ``None`` if the user declined / no
    GUI is available. Never raises — worst case it prints the instructions.
    """
    text = _missing_checkpoint_text(requested)
    if not (want_gui and _can_open_gui()):
        print(text, file=sys.stderr)
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        answer = messagebox.askyesnocancel(
            "No trained model yet",
            text + "\n\nYes = open the training launcher\n"
                   "No = browse for an existing .pt file\n"
                   "Cancel = quit",
            parent=root)
        chosen: Path | None = None
        if answer is True:
            script = PROJECT_DIR / "train.py"
            subprocess.Popen([sys.executable, str(script), "--gui"],
                             cwd=str(PROJECT_DIR))
            print("Training launcher opened — the chat window opens "
                  "automatically when training finishes.")
        elif answer is False:
            picked = filedialog.askopenfilename(
                parent=root, title="Choose a checkpoint",
                initialdir=str(PROJECT_DIR),
                filetypes=[("PyTorch checkpoint", "*.pt *.pth"),
                           ("All files", "*.*")])
            if picked:
                chosen = Path(picked)
        root.destroy()
        return chosen
    except Exception as exc:  # never let the helper itself crash the app
        print(text, file=sys.stderr)
        print(f"(Could not show the dialog: {exc})", file=sys.stderr)
        return None


def main() -> int:
    args = build_parser().parse_args()

    checkpoint = find_checkpoint(args.checkpoint)
    if checkpoint is None:
        checkpoint = _handle_missing_checkpoint(
            args.checkpoint, want_gui=not args.cli)
        if checkpoint is None:
            return 2
    elif Path(args.checkpoint).name == DEFAULT_CHECKPOINT_NAME and \
            checkpoint.name != DEFAULT_CHECKPOINT_NAME:
        print(f"(Default checkpoint missing — using newest model found: "
              f"{checkpoint})")

    settings = {} if args.no_settings else load_settings()

    use_gui = not args.cli
    if use_gui and not _can_open_gui():
        print("(No display — falling back to CLI mode.)\n")
        use_gui = False

    if use_gui:
        launch_gui(checkpoint, settings, qa_default=args.qa)
        return 0
    return run_cli(checkpoint, args, settings)


if __name__ == "__main__":
    sys.exit(main())