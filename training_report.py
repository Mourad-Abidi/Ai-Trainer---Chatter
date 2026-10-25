"""Professional Training Report Window with Graphs, Comparison Tables, AI Recommendations, and Export (Markdown, PDF, GGUF).

Usage:
    python training_report.py [--log training_log.csv]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

# Ensure Tkinter backend for window
if matplotlib.get_backend().lower() == "agg":
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    CANVAS_TK_AVAILABLE = True
except ImportError:
    FigureCanvasTkAgg = None
    CANVAS_TK_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False
    tk = None
    ttk = None

from export_gguf import export_checkpoint_to_gguf
from plot_results import load_log, run_labels

LOG_FILE = "training_log.csv"
GUI_BG = "#0a0c11"
GUI_PANEL = "#11141b"
GUI_PANEL2 = "#1a1e29"
GUI_ACCENT = "#4cc9f0"
GUI_TEXT = "#e8ebf1"
GUI_DIM = "#8d96a6"
GUI_BORDER = "#232733"
GUI_GOOD = "#86efac"
GUI_WARN = "#ffb703"
GUI_BAD = "#f87171"
GUI_GOLD = "#ffd60a"


def analyze_training_data(df: pd.DataFrame) -> dict[str, Any]:
    """Extract metrics, session comparisons, and AI suggestions from dataframe."""
    runs = list(dict.fromkeys(df["run_id"]))
    labels = run_labels(df)

    sessions_summary = []
    for r in runs:
        sub = df[df["run_id"] == r]
        vals = sub.dropna(subset=["val_loss"])
        sess_num = sub["session"].dropna().max() if "session" in sub and sub["session"].notna().any() else 1
        sess_num = int(sess_num) if not pd.isna(sess_num) else 1

        steps = int(sub["step"].max()) if not sub.empty else 0
        best_val = float(vals["val_loss"].min()) if not vals.empty else float("nan")
        last_val = float(vals["val_loss"].iloc[-1]) if not vals.empty else float("nan")
        first_val = float(vals["val_loss"].iloc[0]) if not vals.empty else float("nan")

        train_loss = float(sub["train_loss"].dropna().iloc[-1]) if "train_loss" in sub and sub["train_loss"].notna().any() else float("nan")
        gap = float(sub["gap"].dropna().iloc[-1]) if "gap" in sub and sub["gap"].notna().any() else (last_val - train_loss)

        tps = sub["tokens_per_sec"].dropna()
        speed = float(tps.median()) if not tps.empty else 0.0

        elapsed = float(sub["elapsed"].max()) if "elapsed" in sub and sub["elapsed"].notna().any() else 0.0

        sessions_summary.append({
            "run_id": r,
            "label": labels.get(r, f"session {sess_num}"),
            "session": sess_num,
            "total_steps": steps,
            "first_val_loss": first_val,
            "best_val_loss": best_val,
            "last_val_loss": last_val,
            "train_loss": train_loss,
            "gap": gap,
            "perplexity": math.exp(min(best_val, 20)) if not math.isnan(best_val) else float("nan"),
            "bpc": best_val / math.log(2) if not math.isnan(best_val) else float("nan"),
            "speed": speed,
            "elapsed_min": elapsed / 60.0,
        })

    latest = sessions_summary[-1] if sessions_summary else {}
    older = sessions_summary[:-1] if len(sessions_summary) > 1 else []

    # Generate Smart Suggestions
    suggestions = []
    if latest:
        gap = latest.get("gap", 0.0)
        best_val = latest.get("best_val_loss", float("nan"))
        first_val = latest.get("first_val_loss", float("nan"))

        if not math.isnan(gap) and gap > 0.15:
            suggestions.append(
                "⚠️ **Overfitting Detected**: Validation loss is higher than training loss by > 0.15. "
                "Consider increasing dropout (e.g. 0.2), adding label smoothing (0.05), or adding more training data."
            )
        elif not math.isnan(gap) and gap <= 0.05:
            suggestions.append(
                "✅ **Healthy Fit**: Overfitting gap is minimal (< 0.05). Model generalizes very well to unseen text."
            )

        if not math.isnan(best_val) and not math.isnan(first_val) and first_val > 0:
            impr = (first_val - best_val) / first_val * 100.0
            if impr > 5.0:
                suggestions.append(
                    f"📈 **Significant Improvement**: Validation loss improved by {impr:.1f}% this session. "
                    "Recommend running another session to continue learning."
                )
            elif impr < 0.5:
                suggestions.append(
                    "⏸️ **Loss Plateau**: Validation loss has stabilized. "
                    "Try reducing the learning rate (e.g., 0.0005) or introducing new training dataset files."
                )

        if len(older) > 0:
            prev = older[-1]
            prev_best = prev.get("best_val_loss", float("nan"))
            if not math.isnan(best_val) and not math.isnan(prev_best):
                diff = best_val - prev_best
                if diff < 0:
                    suggestions.append(
                        f"🎉 **Session Progress**: Last session achieved lower validation loss ({best_val:.4f}) than prior session ({prev_best:.4f})."
                    )
                else:
                    suggestions.append(
                        f"💡 **Session Comparison**: Validation loss ({best_val:.4f}) did not beat previous best ({prev_best:.4f}). "
                        "Check learning rate or data diversity."
                    )

        if not suggestions:
            suggestions.append(
                "💡 **Next Step**: Continue training to lower validation loss. Ensure data quality remains high."
            )

    return {
        "latest": latest,
        "older": older,
        "all_sessions": sessions_summary,
        "suggestions": suggestions,
    }


def export_report_markdown(analysis: dict[str, Any], output_path: str = "training_report.md") -> str:
    """Generate Markdown report file."""
    latest = analysis.get("latest", {})
    all_sess = analysis.get("all_sessions", [])
    suggs = analysis.get("suggestions", [])

    lines = [
        "# 📊 Ai-Trainer — Training Performance Report",
        f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## 🚀 Key Performance Indicators",
        f"- **Best Validation Loss:** `{latest.get('best_val_loss', 0.0):.4f}`",
        f"- **Perplexity:** `{latest.get('perplexity', 0.0):.2f}`",
        f"- **Bits / Char (BPC):** `{latest.get('bpc', 0.0):.3f}`",
        f"- **Total Steps Completed:** `{latest.get('total_steps', 0):,}`",
        f"- **Throughput:** `{latest.get('speed', 0.0):,.0f} chars/sec`",
        f"- **Overfitting Gap:** `{latest.get('gap', 0.0):+.4f}`\n",
        "## 📋 Session Comparison Table",
        "| Session | Steps | Best Val Loss | Train Loss | Overfit Gap | Speed (ch/s) | Duration |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in all_sess:
        lines.append(
            f"| #{s.get('session', 1)} ({s.get('run_id', '')[:6]}) | "
            f"{s.get('total_steps', 0):,} | "
            f"{s.get('best_val_loss', 0.0):.4f} | "
            f"{s.get('train_loss', 0.0):.4f} | "
            f"{s.get('gap', 0.0):+.4f} | "
            f"{s.get('speed', 0.0):,.0f} | "
            f"{s.get('elapsed_min', 0.0):.1f} min |"
        )

    lines.append("\n## 💡 AI Training Recommendations")
    for sg in suggs:
        lines.append(f"- {sg}")

    content = "\n".join(lines)
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path


def export_report_pdf(analysis: dict[str, Any], output_path: str = "training_report.pdf") -> str:
    """Generate PDF report file."""
    output_p = Path(output_path)
    latest = analysis.get("latest", {})
    all_sess = analysis.get("all_sessions", [])
    suggs = analysis.get("suggestions", [])

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(str(output_p), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor("#11141b"),
            spaceAfter=12
        )
        story.append(Paragraph("Ai-Trainer — Training Performance Report", title_style))
        story.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 14))

        # KPI summary
        kpi_text = (
            f"<b>Best Val Loss:</b> {latest.get('best_val_loss', 0.0):.4f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Perplexity:</b> {latest.get('perplexity', 0.0):.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>BPC:</b> {latest.get('bpc', 0.0):.3f}<br/>"
            f"<b>Total Steps:</b> {latest.get('total_steps', 0):,} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Speed:</b> {latest.get('speed', 0.0):,.0f} chars/sec &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Overfit Gap:</b> {latest.get('gap', 0.0):+.4f}"
        )
        story.append(Paragraph(kpi_text, styles['Normal']))
        story.append(Spacer(1, 16))

        # Table
        table_data = [["Session", "Steps", "Best Val Loss", "Train Loss", "Gap", "Speed (ch/s)"]]
        for s in all_sess:
            table_data.append([
                f"#{s.get('session', 1)}",
                f"{s.get('total_steps', 0):,}",
                f"{s.get('best_val_loss', 0.0):.4f}",
                f"{s.get('train_loss', 0.0):.4f}",
                f"{s.get('gap', 0.0):+.4f}",
                f"{s.get('speed', 0.0):,.0f}"
            ])
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11141b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#232733')),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 16))

        story.append(Paragraph("<b>AI Recommendations:</b>", styles['Heading2']))
        for sg in suggs:
            clean_sg = sg.replace("**", "<b>").replace("**", "</b>")
            story.append(Paragraph(f"• {clean_sg}", styles['Normal']))
            story.append(Spacer(1, 4))

        doc.build(story)
        return str(output_p)
    except ImportError:
        # Fallback to Markdown format if ReportLab is absent
        md_file = export_report_markdown(analysis, output_path.replace(".pdf", ".md"))
        return md_file


class TrainingReportWindow:
    """Tkinter Window for Professional Training Report."""

    def __init__(self, root=None, log_file: str = LOG_FILE):
        if root is None:
            self.root = tk.Tk()
            self.is_owner = True
        else:
            self.root = tk.Toplevel(root)
            self.is_owner = False

        self.root.title("Ai-Trainer — Professional Training Report")
        self.root.geometry("1100x800")
        self.root.configure(bg=GUI_BG)

        self.log_file = log_file
        self.df = load_log(log_file, all_runs=True)
        if self.df is None:
            messagebox.showerror("Error", f"No valid training data found in {log_file}")
            return

        self.analysis = analyze_training_data(self.df)
        self._build_ui()

    def _build_ui(self):
        # 1. Top Header Bar
        header = tk.Frame(self.root, bg=GUI_PANEL, height=60)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="📊 Ai-Trainer — Executive Performance Report",
            bg=GUI_PANEL,
            fg=GUI_TEXT,
            font=("TkDefaultFont", 14, "bold"),
            padx=18
        ).pack(side=tk.LEFT, pady=16)

        # Action Buttons
        btn_frame = tk.Frame(header, bg=GUI_PANEL)
        btn_frame.pack(side=tk.RIGHT, padx=18, pady=12)

        ttk.Button(btn_frame, text="📝 Export Markdown", command=self._export_md).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="📄 Export PDF", command=self._export_pdf).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="📦 Export GGUF", command=self._export_gguf).pack(side=tk.LEFT, padx=4)

        # Main Scrollable Body
        container = tk.Frame(self.root, bg=GUI_BG)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=GUI_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas, bg=GUI_BG)

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=12)

        # 2. KPI Summary Cards
        kpi_frame = tk.Frame(scroll_content, bg=GUI_BG)
        kpi_frame.pack(fill=tk.X, pady=(0, 16))

        latest = self.analysis.get("latest", {})
        cards_data = [
            ("BEST VAL LOSS", f"{latest.get('best_val_loss', 0.0):.4f}", GUI_GOLD),
            ("PERPLEXITY", f"{latest.get('perplexity', 0.0):.2f}", GUI_ACCENT),
            ("BITS / CHAR", f"{latest.get('bpc', 0.0):.3f}", GUI_ACCENT),
            ("OVERFIT GAP", f"{latest.get('gap', 0.0):+.4f}", GUI_GOOD if latest.get('gap', 0) < 0.1 else GUI_WARN),
            ("TOTAL STEPS", f"{latest.get('total_steps', 0):,}", GUI_TEXT),
            ("THROUGHPUT", f"{latest.get('speed', 0.0):,.0f} c/s", GUI_TEXT),
        ]

        for i, (title, val, color) in enumerate(cards_data):
            card = tk.Frame(kpi_frame, bg=GUI_PANEL, highlightthickness=1, highlightbackground=GUI_BORDER)
            card.grid(row=0, column=i, padx=6, pady=4, sticky="nsew")
            kpi_frame.grid_columnconfigure(i, weight=1)

            tk.Label(card, text=title, bg=GUI_PANEL, fg=GUI_DIM, font=("TkDefaultFont", 8, "bold")).pack(padx=10, pady=(8, 2))
            tk.Label(card, text=val, bg=GUI_PANEL, fg=color, font=("TkDefaultFont", 14, "bold")).pack(padx=10, pady=(0, 8))

        # 3. Session Comparison Table
        tbl_frame = tk.LabelFrame(scroll_content, text=" 📋 Session Comparison Table ", bg=GUI_PANEL, fg=GUI_ACCENT, font=("TkDefaultFont", 10, "bold"), padx=12, pady=10)
        tbl_frame.pack(fill=tk.X, pady=(0, 16))

        cols = ("Session", "Run ID", "Total Steps", "Best Val Loss", "Train Loss", "Overfit Gap", "Speed (c/s)", "Duration")
        tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=min(6, len(self.analysis.get("all_sessions", []))))
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=110)

        for s in self.analysis.get("all_sessions", []):
            tree.insert("", tk.END, values=(
                f"#{s.get('session', 1)}",
                str(s.get('run_id', ''))[:8],
                f"{s.get('total_steps', 0):,}",
                f"{s.get('best_val_loss', 0.0):.4f}",
                f"{s.get('train_loss', 0.0):.4f}",
                f"{s.get('gap', 0.0):+.4f}",
                f"{s.get('speed', 0.0):,.0f}",
                f"{s.get('elapsed_min', 0.0):.1f} m",
            ))
        tree.pack(fill=tk.X)

        # 4. Embedded Charts
        chart_frame = tk.LabelFrame(scroll_content, text=" 📈 Loss & Metrics Curves ", bg=GUI_PANEL, fg=GUI_ACCENT, font=("TkDefaultFont", 10, "bold"), padx=8, pady=8)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 16))

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=GUI_PANEL)
        fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25, top=0.88, bottom=0.18)

        # Loss Chart
        ax1 = axes[0]
        ax1.set_facecolor(GUI_BG)
        ax1.set_title("Validation vs Train Loss", color=GUI_TEXT, fontsize=10)
        ax1.plot(self.df["step"], self.df["val_loss"], color=GUI_ACCENT, label="Val Loss", linewidth=1.8)
        if "train_loss" in self.df:
            ax1.plot(self.df["step"], self.df["train_loss"], color=GUI_WARN, linestyle="--", label="Train Loss", linewidth=1.2)
        ax1.tick_params(colors=GUI_DIM, labelsize=8)
        ax1.grid(True, color=GUI_BORDER, linewidth=0.5)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), facecolor=GUI_PANEL, labelcolor=GUI_TEXT, fontsize=8)

        # Gap Chart
        ax2 = axes[1]
        ax2.set_facecolor(GUI_BG)
        ax2.set_title("Overfitting Gap", color=GUI_TEXT, fontsize=10)
        if "gap" in self.df:
            ax2.plot(self.df["step"], self.df["gap"], color=GUI_GOLD, linewidth=1.5)
            ax2.axhline(0, color=GUI_DIM, linestyle=":", linewidth=0.8)
        ax2.tick_params(colors=GUI_DIM, labelsize=8)
        ax2.grid(True, color=GUI_BORDER, linewidth=0.5)

        chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 5. Smart AI Recommendations
        sugg_frame = tk.LabelFrame(scroll_content, text=" 💡 Smart AI Training Recommendations ", bg=GUI_PANEL, fg=GUI_GOLD, font=("TkDefaultFont", 10, "bold"), padx=12, pady=10)
        sugg_frame.pack(fill=tk.X, pady=(0, 16))

        for sg in self.analysis.get("suggestions", []):
            row = tk.Frame(sugg_frame, bg=GUI_PANEL)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text="•", bg=GUI_PANEL, fg=GUI_ACCENT, font=("TkDefaultFont", 12, "bold")).pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=sg.replace("**", ""), bg=GUI_PANEL, fg=GUI_TEXT, font=("TkDefaultFont", 10), wraplength=950, justify="left").pack(side=tk.LEFT, fill=tk.X)

    def _export_md(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
            initialfile="training_report.md"
        )
        if file_path:
            export_report_markdown(self.analysis, file_path)
            messagebox.showinfo("Export Success", f"Report exported to Markdown:\n{file_path}")

    def _export_pdf(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialfile="training_report.pdf"
        )
        if file_path:
            out_f = export_report_pdf(self.analysis, file_path)
            messagebox.showinfo("Export Success", f"Report exported to:\n{out_f}")

    def _export_gguf(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            filetypes=[("GGUF Model", "*.gguf"), ("All files", "*.*")],
            initialfile="ai_trainer_model.gguf"
        )
        if file_path:
            try:
                out_f = export_checkpoint_to_gguf("ai_trainer---chatter.pt", file_path)
                messagebox.showinfo("Export Success", f"GGUF Model exported successfully:\n{out_f}")
            except Exception as exc:
                messagebox.showerror("Export Failed", f"Could not export GGUF model: {exc}")

    def show(self):
        if self.is_owner:
            self.root.mainloop()


def show_report_window(root=None, log_file: str = LOG_FILE):
    """EntryPoint for displaying the training report window."""
    win = TrainingReportWindow(root, log_file)
    win.show()


def main():
    parser = argparse.ArgumentParser(description="Generate Ai-Trainer Report Window")
    parser.add_argument("--log", default=LOG_FILE, help="Path to training log CSV")
    args = parser.parse_args()

    show_report_window(log_file=args.log)


if __name__ == "__main__":
    main()
