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
NUMERIC = ["step", "elapsed", "train_loss", "val_loss", "lr", "grad_norm", "tokens_per_sec"]


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
    df = df.dropna(subset=["step"])
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
            if key == "train_loss" and len(sub) > 8:
                ax.plot(sub["step"], sub[key], color=color, linewidth=0.7, alpha=0.35)
                ax.plot(sub["step"], sub[key].rolling(8, min_periods=1).mean(), color=color,
                        linewidth=1.8, label=run)
            else:
                ax.plot(sub["step"], sub[key], color=color, linewidth=1.6,
                        marker="o" if is_val else None, markersize=3, label=run)
            if key == "val_loss":
                best = sub.loc[sub[key].idxmin()]
                ax.scatter([best["step"]], [best[key]], marker="*", s=130, color="#ffd60a", zorder=5)
                ax.annotate(f"best {best[key]:.3f}", (best["step"], best[key]), color="#ffd60a",
                            fontsize=8, xytext=(6, 8), textcoords="offset points")
            if key == "gap":
                ax.axhline(0, color=GRID, linewidth=1)
        if multi and key == "train_loss":
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, title="run", title_fontsize=7)

    last_run = runs[-1]
    cur = df[df["run_id"] == last_run]
    vals = cur.dropna(subset=["val_loss"])
    row = cur.iloc[-1]
    summary = f"run {last_run}  |  step {int(row['step']):,}  |  train {row['train_loss']:.4f}"
    if not vals.empty:
        v = vals.iloc[-1]["val_loss"]
        summary += f"  |  val {v:.4f}  |  best val {vals['val_loss'].min():.4f}  |  ppl {math.exp(min(v, 20)):.2f}"
    summary += f"  |  {row['elapsed'] / 60:.1f} min elapsed"
    fig.suptitle(summary, color="#dddddd", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))


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


def run_live(log, run_id, interval, save_png=False):
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
        draw(fig, axes, df)
        if save_png:
            fig.savefig(OUTPUT_IMAGE_PATH, dpi=100, facecolor=fig.get_facecolor())

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
        run_live(args.log, args.run_id, args.interval, args.save_png)
    elif save_static(args.log, args.out, args.run_id):
        print(f"Saved dashboard to '{args.out}'")


if __name__ == "__main__":
    main()