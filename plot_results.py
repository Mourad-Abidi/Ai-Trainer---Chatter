"""Training dashboard.

  python plot_results.py --live      # popup window, refreshes while training runs
  python plot_results.py             # save a static PNG from training_log.csv

train.py opens the live window automatically (use --no-live to disable).

What you see
------------
* A strip of headline numbers (total steps, best score, perplexity, ...).
* Ten graphs. Click "Expand" to pop out a graph in a new separate window, or
  click "?" on any graph for a plain-English explanation.
* Training is CUMULATIVE: every session of the same model continues the same
  curves. Dotted vertical lines mark where a new session started.
"""
from __future__ import annotations

import argparse
import math
import os
import textwrap
import time

import matplotlib
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False
    tk = None
    ttk = None

LOG_FILE = "training_log.csv"
OUTPUT_IMAGE_PATH = "training_dashboard.png"

BG, CARD, GRID, TEXT = "#0a0c11", "#11141b", "#232733", "#8d96a6"
HI, ACCENT = "#e8ebf1", "#4cc9f0"
GOOD, WARN, BAD, GOLD = "#86efac", "#ffb703", "#f87171", "#ffd60a"
PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffb703", "#9d4edd", "#ff7b00",
           "#2ec4b6", "#e63946"]

# key, title, y-label, plain-English explanation (shown by the "?" buttons)
PANELS = [
    ("train_loss", "Training loss", "cross-entropy",
     "How many mistakes the AI makes on the text it is STUDYING. Lower is "
     "better.\n\nThe faint line is the raw value, the bright line is "
     "smoothed. It should fall fast at first, then slowly.\n\nWild "
     "zig-zags = the learning rate is probably too high."),
    ("val_loss", "Validation loss  (the one that matters)", "cross-entropy",
     "Mistakes on text the AI has NEVER seen - a practice exam. Lower = "
     "smarter.\n\nSolid line: exam. Dashed line: the same test on the study "
     "text. The star marks the best score; that version is the one saved.\n\n"
     "If the solid line climbs while the dashed line keeps falling, the AI "
     "is memorising instead of understanding."),
    ("val_ppl", "Validation perplexity", "perplexity",
     "'How many letters is the AI torn between?' on average. 1 would be "
     "perfect. 3 means it is guessing among about 3 likely letters.\n\n"
     "Lower is better. A log scale is used when the range is huge."),
    ("val_bpc", "Validation bits per character", "bits / char",
     "The same score expressed as yes/no questions needed to guess the next "
     "letter.\n\nLower is better. Good English models reach roughly "
     "1.0 - 1.5."),
    ("gap", "Overfitting gap  (exam - study)", "loss gap",
     "Exam score minus study score, measured the same way.\n\nNear 0 = "
     "healthy. Growing steadily = the AI is memorising: give it more text "
     "or raise dropout.\n\nSlightly negative early on is normal."),
    ("lr", "Learning rate", "lr",
     "How big each learning 'nudge' is. It ramps up (warm-up), then glides "
     "down for a gentle landing.\n\nA continued session restarts a smaller "
     "cycle. Dotted vertical lines mark the start of each session."),
    ("grad_norm", "Gradient norm (pre-clip)", "norm",
     "How hard the AI was pushed to change at each step, before the safety "
     "limit.\n\nSmooth and slowly shrinking is healthy. Huge spikes = "
     "unstable: lower the learning rate."),
    ("tokens_per_sec", "Throughput", "chars / sec",
     "Training speed: letters studied per second. Higher = faster.\n\nA "
     "sudden drop usually means another program is using your computer."),
]
PANEL_KEYS = [p[0] for p in PANELS]

# Noisy per-step series get a faint raw trace plus a smoothed line.
SMOOTHED = {"train_loss", "grad_norm", "tokens_per_sec"}
SMOOTH_WINDOW = 8
NUMERIC = ["step", "elapsed", "train_loss", "val_loss", "lr", "grad_norm",
           "tokens_per_sec", "train_clean", "session"]
REQUIRED = {"run_id", "step", "train_loss", "val_loss"}
LIVE_SECONDS = 120          # no new log row for this long -> "idle"


# ── formatting ───────────────────────────────────────────────────────────
def _human(v, _pos=None):
    """1500 -> 1.5k, 2_000_000 -> 2M (compact axis labels)."""
    a = abs(v)
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{v / div:.3g}{suffix}"
    return f"{v:.3g}"


HUMAN = FuncFormatter(_human)


def _fmt_age(seconds: float) -> str:
    if seconds < 10:
        return "just now"
    if seconds < 90:
        return f"{int(seconds // 10 * 10)}s ago"
    if seconds < 5400:
        return f"{int(round(seconds / 60))} min ago"
    return f"{seconds / 3600:.1f} h ago"


# ── data ─────────────────────────────────────────────────────────────────
def _session_of(sub: pd.DataFrame):
    s = sub["session"].dropna()
    return int(s.max()) if len(s) else None


def _latest_lineage(df: pd.DataFrame, max_runs: int) -> pd.DataFrame:
    """Keep the newest run plus the earlier sessions it continued from."""
    runs = list(dict.fromkeys(df["run_id"]))
    sess = {r: _session_of(df[df["run_id"] == r]) or 1 for r in runs}
    i = len(runs) - 1
    keep = [runs[i]]
    while i > 0 and sess[runs[i]] > 1:      # session 2+ continued a parent
        i -= 1
        keep.append(runs[i])
    keep = list(reversed(keep))[-max_runs:]
    return df[df["run_id"].isin(keep)]


def load_log(path, run_id=None, all_runs=False, max_runs=8):
    """Read the CSV log tolerantly. Returns ``None`` if there is nothing to
    draw yet (missing / empty / half-written file)."""
    if pd is None:
        import csv
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                reader = list(csv.DictReader(f))
            if not reader:
                return None
            import pandas as pdd
            df = pdd.DataFrame(reader)
        except Exception:
            return None
    else:
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except (OSError, ValueError, Exception):
            return None
    if not REQUIRED.issubset(df.columns):
        return None
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df \
            else np.nan
    df["run_id"] = df["run_id"].astype(str)
    df = df.dropna(subset=["step"])
    if df.empty:
        return None

    order = {r: i for i, r in enumerate(dict.fromkeys(df["run_id"]))}
    df["_ord"] = df["run_id"].map(order)
    df = df.sort_values(["_ord", "step"], kind="stable").reset_index(drop=True)
    if run_id:
        df = df[df["run_id"] == str(run_id)]
    elif not all_runs:
        df = _latest_lineage(df, max_runs)
    if df.empty:
        return None
    df = df.copy()
    df["val_ppl"] = np.exp(df["val_loss"].clip(upper=20))
    df["val_bpc"] = df["val_loss"] / math.log(2)
    smooth_train = df.groupby("run_id")["train_loss"].transform(
        lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1).mean())
    df["gap"] = df["val_loss"] - df["train_clean"].fillna(smooth_train)
    return df


def run_labels(df: pd.DataFrame) -> dict:
    labels = {}
    for r in dict.fromkeys(df["run_id"]):
        s = _session_of(df[df["run_id"] == r])
        labels[r] = f"session {s}" if s else f"run {r[:6]}"
    return labels


# ── figure construction ──────────────────────────────────────────────────
def make_figure(plt, live=False):
    fig = plt.figure(figsize=(15.0, 9.5) if live else (16.0, 10.0),
                     facecolor=BG)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.45, 1, 1], hspace=0.52,
                          wspace=0.28, left=0.04, right=0.96,
                          top=0.91, bottom=0.065)
    fig._kpi = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1 + i // 4, i % 4]) for i in range(8)]
    fig._artists = {}
    fig._help_open = set()
    fig._df = None
    fig._sig = None
    return fig, axes


def _style(ax, title, ylabel, xlabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=HI, fontsize=10.0, pad=9, loc="left",
                 fontweight="bold")
    ax.set_xlabel(xlabel, color=TEXT, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.set_major_formatter(HUMAN)


def _set_text(fig, name, x, y, text, **kw):
    old = fig._artists.pop(name, None)
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    fig._artists[name] = fig.text(x, y, text, **kw)
    return fig._artists[name]


def _fit_loss_ylim(ax, arrays):
    parts = [a[np.isfinite(a)] for a in arrays if len(a)]
    if not parts:
        return
    vals = np.concatenate(parts)
    if vals.size < 12:
        return
    lo, hi = float(vals.min()), float(np.percentile(vals, 97))
    if hi - lo < 1e-9:
        return
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _kpi_cards(ax, cards):
    ax.clear()
    ax.set_facecolor(BG)
    ax.axis("off")
    n = len(cards)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    for i, (title, value, sub, color) in enumerate(cards):
        ax.add_patch(FancyBboxPatch(
            (i + 0.04, 0.06), 0.92, 0.88,
            boxstyle="round,pad=0,rounding_size=0.06",
            transform=ax.transData, facecolor=CARD, edgecolor=GRID,
            linewidth=1.0, mutation_aspect=0.25))
        ax.text(i + 0.10, 0.80, title, color=TEXT, fontsize=8,
                fontweight="bold", va="center")
        ax.text(i + 0.10, 0.48, value, color=color, fontsize=17,
                fontweight="bold", va="center")
        ax.text(i + 0.10, 0.17, sub, color=TEXT, fontsize=8, va="center")


# ── drawing ──────────────────────────────────────────────────────────────
def _status(vals):
    if len(vals) < 3:
        return "● warming up — waiting for more test results", TEXT
    v = vals["val_loss"].to_numpy()
    if v[-1] >= v[-2] >= v[-3] and v[-1] > v.min() * 1.005:
        return ("⚠ getting worse on unseen text — possible overfitting; "
                "the best version is already saved"), WARN
    if len(v) >= 5 and abs(v[-1] - v[-5]) < 0.002 * max(v[-5], 1e-9):
        return ("● plateau — try more text, a lower learning rate, "
                "or a bigger model"), TEXT
    return "● improving — validation loss is still dropping", GOOD


def _help_box(text):
    paras = text.split("\n\n")
    return "\n\n".join(textwrap.fill(p, 44) for p in paras) \
        + "\n\n(click here to close)"


def draw(fig, axes, df, interactive=False, file_age=None):
    runs = list(dict.fromkeys(df["run_id"]))
    labels = run_labels(df)
    multi = len(runs) > 1
    colors = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    xlabel = "total steps (all sessions)" if multi else "step"
    help_open = getattr(fig, "_help_open", set())
    starts = [(r, float(df.loc[df["run_id"] == r, "step"].min()))
              for r in runs]

    for ax, (key, title, ylabel, help_text) in zip(axes, PANELS):
        ax.clear()
        _style(ax, title, ylabel, xlabel)
        zoom = []
        for r in runs:
            sub = df[df["run_id"] == r][["step", key]].dropna()
            if key == "tokens_per_sec" and len(sub) > 1:
                sub = sub.iloc[1:]
            if sub.empty:
                continue
            color = colors[r]
            is_val = key.startswith("val_") or key == "gap"
            if key in SMOOTHED and len(sub) > SMOOTH_WINDOW:
                ax.plot(sub["step"], sub[key], color=color, linewidth=0.7,
                        alpha=0.28)
                ax.plot(sub["step"], sub[key].rolling(
                    SMOOTH_WINDOW, min_periods=1).mean(), color=color,
                    linewidth=1.9, label=labels[r])
            else:
                ax.plot(sub["step"], sub[key], color=color, linewidth=1.7,
                        marker="o" if is_val else None, markersize=3.2,
                        label=labels[r])
            if key == "train_loss":
                zoom.append(sub[key].to_numpy())
            if key == "val_loss":
                zoom.append(sub[key].to_numpy())
                clean = df[df["run_id"] == r][["step", "train_clean"]].dropna()
                if len(clean) > 1:
                    ax.plot(clean["step"], clean["train_clean"], color=color,
                            linewidth=1.1, linestyle="--", alpha=0.55,
                            zorder=1)
                    zoom.append(clean["train_clean"].to_numpy())
                best = sub.loc[sub[key].idxmin()]
                ax.axvline(best["step"], color=GOLD, linewidth=0.7,
                           linestyle=":", alpha=0.35)
                ax.scatter([best["step"]], [best[key]], marker="*", s=120,
                           color=GOLD, zorder=5)
                if r == runs[-1]:
                    ax.annotate(f"best {best[key]:.3f}",
                                (best["step"], best[key]), color=GOLD,
                                fontsize=8.5, fontweight="bold",
                                xytext=(7, 9), textcoords="offset points")
            if key == "gap":
                ax.axhline(0, color="#555b6a", linewidth=1, linestyle="--")
                ax.fill_between(sub["step"], 0, sub[key], color=color,
                                alpha=0.12)
        if key in ("train_loss", "val_loss"):
            _fit_loss_ylim(ax, zoom)
        if key == "val_ppl":
            p = df["val_ppl"].dropna()
            if len(p) > 1 and p.min() > 0 and p.max() / p.min() > 20:
                ax.set_yscale("log")
        if key == "tokens_per_sec":
            ax.yaxis.set_major_formatter(HUMAN)
            ax.set_ylim(bottom=0)
        elif key == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.yaxis.get_offset_text().set_color(TEXT)
        if multi:
            for _r, x0 in starts[1:]:
                ax.axvline(x0, color="#5b6275", linewidth=0.9,
                           linestyle=(0, (2, 3)), zorder=1)

        # Legend positioned cleanly inside subplot boundary (above/upper right)
        if key == "val_loss" and any(ax.lines):
            h1, = ax.plot([], [], color=TEXT, linewidth=1.7, marker="o",
                          markersize=3.2)
            h2, = ax.plot([], [], color=TEXT, linestyle="--", linewidth=1.1,
                          alpha=0.7)
            leg = ax.legend([h1, h2], ["val (exam)", "train (test)"],
                            fontsize=7.0, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT,
                            loc="lower right", bbox_to_anchor=(1.0, 1.01))
            leg.get_frame().set_alpha(0.85)
        elif multi and key == "train_loss" and any(ax.lines):
            leg = ax.legend(fontsize=7.0, facecolor=CARD, edgecolor=GRID,
                            labelcolor=TEXT, loc="lower right", bbox_to_anchor=(1.0, 1.01))
            leg.get_frame().set_alpha(0.85)

        if not any(l.get_xdata().size for l in ax.lines):
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT, fontsize=9,
                    alpha=0.6)

        if interactive:
            # Expand button on header
            exp = ax.text(0.80, 1.035, " ⤢ Expand ", transform=ax.transAxes,
                          ha="right", va="bottom", fontsize=8.0,
                          fontweight="bold", color=ACCENT, picker=True,
                          bbox=dict(boxstyle="round,pad=0.2", fc=CARD,
                                    ec=ACCENT, lw=0.8))
            exp._expand_key = key

            q = ax.text(1.0, 1.035, " ? ", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=9.5,
                        fontweight="bold", color=ACCENT, picker=True,
                        bbox=dict(boxstyle="round,pad=0.25", fc=CARD,
                                  ec=ACCENT, lw=0.9))
            q._help_key = key
            if key in help_open:
                box = ax.text(
                    0.5, 0.5, _help_box(help_text),
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.6, color=HI, zorder=30, picker=True,
                    bbox=dict(boxstyle="round,pad=0.9", fc="#0d1017",
                              ec=ACCENT, lw=1.2, alpha=0.97))
                box._help_key = key

    # ── headline cards ───────────────────────────────────────────────
    last = runs[-1]
    cur = df[df["run_id"] == last]
    vals = cur.dropna(subset=["val_loss"])
    total = int(cur["step"].max())
    sess = _session_of(cur)
    cards = [("TOTAL STEPS", f"{total:,}",
              f"{len(runs)} session{'s' if len(runs) != 1 else ''} shown",
              HI),
             ("SESSION", f"#{sess}" if sess else "—",
              f"{cur['elapsed'].max() / 60:.1f} min elapsed"
              if cur["elapsed"].notna().any() else "", HI)]
    status_text, status_color = _status(vals)
    if not vals.empty:
        best = float(vals["val_loss"].min())
        last_v = float(vals["val_loss"].iloc[-1])
        first = vals.iloc[0]
        started_from_model = bool(pd.isna(first["train_loss"]))
        delta = (f"{best - float(first['val_loss']):+.3f} this session"
                 if started_from_model and len(vals) > 1 else
                 f"latest {last_v:.3f}")
        gap_row = vals.dropna(subset=["gap"])
        gap = float(gap_row["gap"].iloc[-1]) if not gap_row.empty else None
        k = min(5, len(vals) - 1)
        pace = None
        if k >= 1:
            dv = float(vals["val_loss"].iloc[-1]
                       - vals["val_loss"].iloc[-1 - k])
            ds = float(vals["step"].iloc[-1] - vals["step"].iloc[-1 - k])
            if ds > 0:
                pace = dv / ds * 1000.0
        rel = (gap / max(last_v, 1e-9)) if gap is not None else 0.0
        cards += [
            ("BEST VAL LOSS", f"{best:.4f}", delta, GOLD),
            ("PERPLEXITY", f"{math.exp(min(best, 20)):.2f}",
             f"at best · latest {math.exp(min(last_v, 20)):.2f}", ACCENT),
            ("BITS / CHAR", f"{best / math.log(2):.3f}",
             "at best score", ACCENT),
            ("OVERFIT GAP", f"{gap:+.3f}" if gap is not None else "—",
             "healthy" if rel < 0.10 else "watch it" if rel < 0.25
             else "overfitting",
             GOOD if rel < 0.10 else WARN if rel < 0.25 else BAD),
            ("LEARNING PACE", f"{pace:+.3f}" if pace is not None else "—",
             "val loss per 1k steps",
             GOOD if (pace is not None and pace < 0) else WARN),
        ]
    else:
        cards += [("BEST VAL LOSS", "—", "waiting for first test", TEXT)]
    tps = cur["tokens_per_sec"].iloc[1:].dropna()
    cards.append(("SPEED", _human(tps.median()) if not tps.empty else "—",
                  "chars / sec (median)", HI))
    _kpi_cards(fig._kpi, cards)

    # ── header & footer ──────────────────────────────────────────────
    _set_text(fig, "title", 0.012, 0.972,
              "◆  Ai-Trainer · training dashboard", color=HI, fontsize=14,
              fontweight="bold", ha="left", va="center")
    if interactive and file_age is not None:
        live = file_age <= LIVE_SECONDS
        badge = (f"● LIVE · updated {_fmt_age(file_age)}" if live
                 else f"○ idle · last update {_fmt_age(file_age)}")
        bcolor = GOOD if live else TEXT
    else:
        badge = "snapshot · " + time.strftime("%Y-%m-%d %H:%M")
        bcolor = TEXT
    _set_text(fig, "live", 0.96, 0.972, badge, color=bcolor, fontsize=10.5,
              fontweight="bold", ha="right", va="center")
    if interactive:
        g = _set_text(fig, "helpall", 0.96, 0.935,
                      "  ?  explain every graph  ", color=ACCENT, fontsize=9,
                      fontweight="bold", ha="right", va="center", picker=True,
                      bbox=dict(boxstyle="round,pad=0.3", fc=CARD, ec=ACCENT,
                                lw=0.9))
        g._help_key = "__all__"

    parts = []
    if not vals.empty:
        bi = vals["val_loss"].idxmin()
        parts.append(f"best val {vals.loc[bi, 'val_loss']:.4f} @ step "
                     f"{int(vals.loc[bi, 'step']):,}")
        if len(vals) >= 6:
            r5 = vals["val_loss"].iloc[-5:]
            parts.append(f"last 5 tests: {r5.iloc[-1] - r5.iloc[0]:+.4f}")
    if multi:
        parts.append("dotted lines = new session")
    _set_text(fig, "footer", 0.012, 0.018, "   ·   ".join(parts), color=TEXT,
              fontsize=9, ha="left", va="center")
    _set_text(fig, "status", 0.96, 0.018, status_text, color=status_color,
              fontsize=10, fontweight="bold", ha="right", va="center")


# ── interactivity ────────────────────────────────────────────────────────
def popout_graph_window(key: str, df: pd.DataFrame):
    """Open a separate window with an enlarged, interactive view of a graph."""
    if not TK_AVAILABLE or df is None:
        return
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import matplotlib.pyplot as plt
    except ImportError:
        return

    title_map = {p[0]: p[1] for p in PANELS}
    title = title_map.get(key, key)

    top = tk.Toplevel()
    top.title(f"Expanded Graph — {title}")
    top.geometry("900x600")
    top.configure(bg=BG)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    _style(ax, title, "", "step")

    runs = list(dict.fromkeys(df["run_id"]))
    colors = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    labels = run_labels(df)

    for r in runs:
        sub = df[df["run_id"] == r][["step", key]].dropna()
        if not sub.empty:
            ax.plot(sub["step"], sub[key], color=colors[r], label=labels[r], linewidth=2.0)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    if leg:
        leg.get_frame().set_alpha(0.85)

    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, top)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def _on_pick(event):
    fig = event.canvas.figure
    exp_key = getattr(event.artist, "_expand_key", None)
    if exp_key and fig._df is not None:
        popout_graph_window(exp_key, fig._df)
        return

    key = getattr(event.artist, "_help_key", None)
    if key is None:
        return
    opened = fig._help_open
    if key == "__all__":
        if len(opened) >= len(PANEL_KEYS):
            opened.clear()
        else:
            opened.update(PANEL_KEYS)
    elif key in opened:
        opened.discard(key)
    else:
        opened.add(key)
    if fig._df is not None:
        draw(fig, fig._panels, fig._df, interactive=True,
             file_age=getattr(fig, "_age", None))
        fig.canvas.draw_idle()


def _stat_sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size), st.st_mtime
    except OSError:
        return None, None


# ── entry points ─────────────────────────────────────────────────────────
def save_static(log=LOG_FILE, out=OUTPUT_IMAGE_PATH, run_id=None, dpi=150,
                all_runs=False, max_runs=8):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    df = load_log(log, run_id, all_runs, max_runs)
    if df is None:
        print(f"No usable data in '{log}'. Run train.py first.")
        return False
    fig, axes = make_figure(plt)
    draw(fig, axes, df, interactive=False)
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def run_live(log, run_id, interval, save_png=False, out=OUTPUT_IMAGE_PATH,
             all_runs=False, max_runs=8):
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() == "agg":
        print("No GUI backend available (headless?). Use "
              "`python plot_results.py` for a PNG instead.")
        return

    # If Tkinter is available, launch as a scrollable Tkinter App with Collapsible Logs
    if TK_AVAILABLE:
        root = tk.Tk()
        root.title("Ai-Trainer — Live Dashboard & Log Monitor")
        root.geometry("1200x850")
        root.configure(bg=BG)

        # Header Control Bar
        hdr = tk.Frame(root, bg=CARD, height=50)
        hdr.pack(fill=tk.X, side=tk.TOP)

        tk.Label(hdr, text="◆ Ai-Trainer · Live Dashboard", bg=CARD, fg=HI, font=("TkDefaultFont", 12, "bold"), padx=12).pack(side=tk.LEFT, pady=10)

        def open_report():
            try:
                from training_report import show_report_window
                show_report_window(root, log)
            except Exception as exc:
                print(f"Could not open report window: {exc}")

        ttk.Button(hdr, text="📊 Full Report", command=open_report).pack(side=tk.RIGHT, padx=8)

        # Scrollable Frame for Canvas
        container = tk.Frame(root, bg=BG)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas, bg=BG)

        window_id = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        def _on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Matplotlib Figure inside Scroll Content
        fig, axes = make_figure(plt, live=True)
        fig._panels = axes
        fig._age = None
        fig.canvas.mpl_connect("pick_event", _on_pick)

        fig_canvas = FigureCanvasTkAgg(fig, master=scroll_content)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Collapsible Background Log Section at Bottom
        log_frame = tk.LabelFrame(scroll_content, text=" 📜 Background Activity Logs ", bg=CARD, fg=ACCENT, font=("TkDefaultFont", 10, "bold"), padx=10, pady=6)
        log_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        log_visible = tk.BooleanVar(value=False)
        log_text_box = tk.Text(log_frame, bg=BG, fg=HI, height=8, font=("TkFixedFont", 9), wrap=tk.NONE)
        log_text_box.pack_forget()  # Hidden by default

        def toggle_logs():
            if log_visible.get():
                log_text_box.pack_forget()
                log_visible.set(False)
                toggle_btn.configure(text="▼ Show Background Logs")
            else:
                log_text_box.pack(fill=tk.BOTH, expand=True, pady=4)
                log_visible.set(True)
                toggle_btn.configure(text="▲ Hide Background Logs")

        toggle_btn = ttk.Button(log_frame, text="▼ Show Background Logs", command=toggle_logs)
        toggle_btn.pack(side=tk.TOP, anchor="w", pady=2)

        def update():
            try:
                sig, mtime = _stat_sig(log)
                if sig is not None and sig != fig._sig:
                    df = load_log(log, run_id, all_runs, max_runs)
                    if df is not None:
                        fig._sig, fig._df = sig, df
                        fig._age = max(0.0, time.time() - mtime)
                        draw(fig, axes, df, interactive=True, file_age=fig._age)
                        fig.canvas.draw_idle()
                        if save_png:
                            fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())

                        # Update background logs lazily
                        if log_visible.get() and os.path.exists(log):
                            try:
                                with open(log, "r", encoding="utf-8", errors="ignore") as f:
                                    lines = f.readlines()[-100:]
                                log_text_box.configure(state=tk.NORMAL)
                                log_text_box.delete("1.0", tk.END)
                                log_text_box.insert(tk.END, "".join(lines))
                                log_text_box.see(tk.END)
                                log_text_box.configure(state=tk.DISABLED)
                            except Exception:
                                pass
                        return
            except Exception as exc:
                print(f"(dashboard refresh skipped: {exc})")

            root.after(max(250, int(interval * 1000)), update)

        root.after(200, update)
        root.mainloop()
        return

    # Fallback to standard matplotlib popup if Tkinter unavailable
    fig, axes = make_figure(plt, live=True)
    fig._panels = axes
    fig._age = None
    fig.canvas.mpl_connect("pick_event", _on_pick)
    _set_text(fig, "title", 0.012, 0.972,
              "◆  Ai-Trainer · waiting for training data…", color=TEXT,
              fontsize=14, fontweight="bold", ha="left", va="center")
    try:
        fig.canvas.manager.set_window_title("Live training dashboard")
    except Exception:
        pass

    def update():
        try:
            sig, mtime = _stat_sig(log)
            if sig is not None and sig != fig._sig:
                df = load_log(log, run_id, all_runs, max_runs)
                if df is not None:
                    fig._sig, fig._df = sig, df
                    fig._age = max(0.0, time.time() - mtime)
                    draw(fig, axes, df, interactive=True, file_age=fig._age)
                    fig.canvas.draw_idle()
                    if save_png:
                        fig.savefig(out, dpi=100,
                                    facecolor=fig.get_facecolor())
                    return
            if sig is not None and fig._df is not None:
                age = max(0.0, time.time() - mtime)
                live = age <= LIVE_SECONDS
                badge = (f"● LIVE · updated {_fmt_age(age)}" if live
                         else f"○ idle · last update {_fmt_age(age)}")
                t = fig._artists.get("live")
                if t is not None and t.get_text() != badge:
                    t.set_text(badge)
                    t.set_color(GOOD if live else TEXT)
                    fig._age = age
                    fig.canvas.draw_idle()
        except Exception as exc:
            print(f"(dashboard refresh skipped: {exc})")

    timer = fig.canvas.new_timer(interval=max(250, int(interval * 1000)))
    timer.add_callback(update)
    timer.start()
    fig._timer = timer
    update()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=LOG_FILE)
    ap.add_argument("--out", default=OUTPUT_IMAGE_PATH)
    ap.add_argument("--run-id", default=None,
                    help="Only plot this one run")
    ap.add_argument("--all-runs", action="store_true",
                    help="Plot every run in the log")
    ap.add_argument("--max-runs", type=int, default=8,
                    help="Most sessions to draw at once (default 8)")
    ap.add_argument("--live", action="store_true",
                    help="Open a live-updating window")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Live check interval in seconds (default 1)")
    ap.add_argument("--save-png", action="store_true",
                    help="In live mode also keep refreshing the PNG")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.live:
        run_live(args.log, args.run_id, args.interval, args.save_png,
                 args.out, args.all_runs, args.max_runs)
    elif save_static(args.log, args.out, args.run_id, args.dpi,
                     args.all_runs, args.max_runs):
        print(f"Saved dashboard to '{args.out}'")


if __name__ == "__main__":
    main()
