"""Training dashboard.

  python plot_results.py --live      # popup window that refreshes while training runs
  python plot_results.py             # save a static PNG from training_log.csv

train.py opens the live window automatically (use --no-live to disable).
"""
import argparse
import math
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.ticker import FuncFormatter

LOG_FILE = "training_log.csv"
OUTPUT_IMAGE_PATH = "training_dashboard.png"

BG, CARD, GRID, TEXT = "#0b0b0b", "#111111", "#262626", "#9a9a9a"
PALETTE = ["#4cc9f0", "#f72585", "#b5e48c", "#ffb703", "#9d4edd", "#ff7b00", "#2ec4b6", "#e63946"]

# key, title, y-label
PANELS = [
    ("train_loss", "Training loss", "cross-entropy"),
    ("val_loss", "Validation loss", "cross-entropy"),
    ("val_ppl", "Validation perplexity", "perplexity"),
    ("val_bpc", "Validation bits per character", "bits/char"),
    ("gap", "Generalisation gap (val - train)", "loss gap"),
    ("lr", "Learning rate", "lr"),
    ("grad_norm", "Gradient norm (pre-clip)", "norm"),
    ("tokens_per_sec", "Throughput", "chars / sec"),
]
# Noisy per-step series get a faint raw trace plus a smoothed line.
SMOOTHED = {"train_loss", "grad_norm", "tokens_per_sec"}
SMOOTH_WINDOW = 8
NUMERIC = ["step", "elapsed", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec"]


def _human(v, _pos=None):
    """1500 -> 1.5k, 2_000_000 -> 2M (compact axis labels)."""
    a = abs(v)
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{v / div:.3g}{suffix}"
    return f"{v:.3g}"


HUMAN = FuncFormatter(_human)


def load_log(path, run_id=None):
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if not set(NUMERIC + ["run_id"]).issubset(df.columns):
        return None
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["run_id"] = df["run_id"].astype(str)
    df = df.dropna(subset=["step"]).sort_values(["run_id", "step"], kind="stable")
    if run_id:
        df = df[df["run_id"] == str(run_id)]
    if df.empty:
        return None
    df["val_ppl"] = df["val_loss"].clip(upper=20).map(lambda v: math.exp(v) if pd.notna(v) else float("nan"))
    df["val_bpc"] = df["val_loss"] / math.log(2)
    df["gap"] = df["val_loss"] - df["train_loss"]
    return df


def make_figure(plt):
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), facecolor=BG)
    return fig, axes.ravel()


def _style(ax, title, ylabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.set_xlabel("step", color=TEXT, fontsize=8)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.set_major_formatter(HUMAN)


def draw(fig, axes, df):
    runs = list(dict.fromkeys(df["run_id"]))
    multi = len(runs) > 1
    for ax, (key, title, ylabel) in zip(axes, PANELS):
        ax.clear()
        _style(ax, title, ylabel)
        for i, run in enumerate(runs):
            sub = df[df["run_id"] == run][["step", key]].dropna()
            if sub.empty:
                continue
            color = PALETTE[i % len(PALETTE)]
            is_val = key.startswith("val_") or key == "gap"
            if key in SMOOTHED and len(sub) > SMOOTH_WINDOW:
                ax.plot(sub["step"], sub[key], color=color, linewidth=0.7, alpha=0.30)
                ax.plot(sub["step"], sub[key].rolling(SMOOTH_WINDOW, min_periods=1).mean(),
                        color=color, linewidth=1.8, label=run)
            else:
                ax.plot(sub["step"], sub[key], color=color, linewidth=1.6,
                        marker="o" if is_val else None, markersize=3, label=run)
            if key == "val_loss":
                tr = df[df["run_id"] == run][["step", "train_loss"]].dropna()
                if len(tr) > 1:   # train overlay makes over/under-fitting visible at a glance
                    ax.plot(tr["step"], tr["train_loss"].rolling(SMOOTH_WINDOW, min_periods=1).mean(),
                            color=color, linewidth=1.1, linestyle="--", alpha=0.45, zorder=1)
                best = sub.loc[sub[key].idxmin()]
                ax.axvline(best["step"], color="#ffd60a", linewidth=0.8, linestyle=":", alpha=0.5)
                ax.scatter([best["step"]], [best[key]], marker="*", s=130, color="#ffd60a", zorder=5)
                ax.annotate(f"best {best[key]:.3f}", (best["step"], best[key]), color="#ffd60a",
                            fontsize=8, xytext=(6, 8), textcoords="offset points")
            if key == "gap":
                ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
                ax.fill_between(sub["step"], 0, sub[key], color=color, alpha=0.10)
        if key == "val_ppl":
            vals_ppl = df["val_ppl"].dropna()
            if len(vals_ppl) > 1 and vals_ppl.min() > 0 and vals_ppl.max() / vals_ppl.min() > 20:
                ax.set_yscale("log")   # early perplexity is huge; log keeps the end-game visible
        if key == "val_loss" and any(ax.lines):
            h1, = ax.plot([], [], color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6)
            h2, = ax.plot([], [], color=TEXT, linewidth=1.6, marker="o", markersize=3)
            leg = ax.legend([h2, h1], ["validation", "train (smoothed)"], fontsize=7,
                            facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc="upper right")
            leg.get_frame().set_alpha(0.8)
        if key == "tokens_per_sec":
            ax.yaxis.set_major_formatter(HUMAN)
            ax.set_ylim(bottom=0)
        elif key == "lr":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if not any(l.get_xdata().size for l in ax.lines):
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center",
                    va="center", color=TEXT, fontsize=9, alpha=0.6)
        if multi and key == "train_loss":
            leg = ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT,
                            title="run", title_fontsize=7)
            leg.get_title().set_color(TEXT)

    last_run = runs[-1]
    cur = df[df["run_id"] == last_run]
    vals = cur.dropna(subset=["val_loss"])
    row = cur.iloc[-1]
    summary = f"run {last_run}  |  step {int(row['step']):,}  |  train {row['train_loss']:.4f}"
    if not vals.empty:
        v = vals.iloc[-1]["val_loss"]
        summary += f"  |  val {v:.4f}  |  best val {vals['val_loss'].min():.4f}  |  ppl {math.exp(min(v, 20)):.2f}"
    summary += f"  |  {row['elapsed'] / 60:.1f} min elapsed"
    if multi:
        summary += f"  |  {len(runs)} runs"
    fig.suptitle(summary, color="#dddddd", fontsize=13, y=0.995)

    # ── insights footer (last run) ────────────────────────────────────
    parts = []
    if not vals.empty:
        bi = vals["val_loss"].idxmin()
        best_loss, best_step = vals.loc[bi, "val_loss"], int(vals.loc[bi, "step"])
        parts.append(f"best val {best_loss:.4f} @ step {best_step:,}  ({best_loss / math.log(2):.3f} bits/char)")
        if len(vals) >= 6:
            recent = vals["val_loss"].iloc[-5:]
            parts.append(f"last 5 evals: {recent.iloc[-1] - recent.iloc[0]:+.4f}")
        gap_now = vals.iloc[-1]["val_loss"] - vals.iloc[-1]["train_loss"] if pd.notna(vals.iloc[-1]["train_loss"]) else None
        if gap_now is not None:
            parts.append(f"gap {gap_now:+.3f}")
    tps = cur["tokens_per_sec"].iloc[1:].dropna()
    if not tps.empty:
        parts.append(f"median {_human(tps.median())} chars/s")
    status, status_color = "", TEXT
    if len(vals) >= 5:
        v = vals["val_loss"].to_numpy()
        rising = v[-1] >= v[-2] >= v[-3] and v[-1] > v.min() * 1.005
        flat = abs(v[-1] - v[-5]) < 0.002 * max(v[-5], 1e-9)
        if rising:
            status, status_color = "⚠ validation loss rising — possible overfitting; best checkpoint is already saved", "#ffb703"
        elif flat:
            status, status_color = "● plateau — lower LR / more data / larger model may help", "#9a9a9a"
        else:
            status, status_color = "● improving", "#b5e48c"
    footer = getattr(fig, "_footer", None)
    if footer is not None:
        try:
            footer.remove()
        except Exception:
            pass
    text = "   ·   ".join(parts)
    fig._footer = fig.text(0.01, 0.012, text, color=TEXT, fontsize=9, ha="left", va="bottom")
    status_t = getattr(fig, "_status", None)
    if status_t is not None:
        try:
            status_t.remove()
        except Exception:
            pass
    fig._status = fig.text(0.99, 0.012, status, color=status_color, fontsize=9.5,
                           ha="right", va="bottom", fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))


def save_static(log=LOG_FILE, out=OUTPUT_IMAGE_PATH, run_id=None, dpi=150):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    df = load_log(log, run_id)
    if df is None:
        print(f"No usable data in '{log}'. Run train.py first.")
        return False
    fig, axes = make_figure(plt)
    draw(fig, axes, df)
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def run_live(log, run_id, interval, save_png=False, out=OUTPUT_IMAGE_PATH):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if matplotlib.get_backend().lower() == "agg":
        print("No GUI backend available (headless?). Use `python plot_results.py` for a PNG instead.")
        return
    fig, axes = make_figure(plt)
    fig.suptitle("Waiting for training data...", color=TEXT)
    try:
        fig.canvas.manager.set_window_title("Live training dashboard")
    except Exception:
        pass

    def update(_):
        df = load_log(log, run_id)
        if df is None:
            return
        try:
            draw(fig, axes, df)
        except Exception as exc:  # never let one bad frame close the window
            print(f"(dashboard refresh skipped: {exc})")
            return
        if save_png:
            fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())

    anim = FuncAnimation(fig, update, interval=int(interval * 1000), cache_frame_data=False)
    fig._anim = anim  # keep a reference so it isn't garbage collected
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=LOG_FILE)
    ap.add_argument("--out", default=OUTPUT_IMAGE_PATH)
    ap.add_argument("--run-id", default=None, help="Only plot this run (default: all runs)")
    ap.add_argument("--live", action="store_true", help="Open a live-updating window")
    ap.add_argument("--interval", type=float, default=2.0, help="Live refresh seconds")
    ap.add_argument("--save-png", action="store_true", help="In live mode also keep refreshing the PNG")
    args = ap.parse_args()

    if args.live:
        run_live(args.log, args.run_id, args.interval, args.save_png, args.out)
    elif save_static(args.log, args.out, args.run_id):
        print(f"Saved dashboard to '{args.out}'")


if __name__ == "__main__":
    main()