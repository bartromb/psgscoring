#!/usr/bin/env python3
"""
validation_report.py — generate a PDF report from validate_psgipa.py JSON.

Reads the JSON written by validate_psgipa.py (default /tmp/validation_results.json)
and produces a 6-page PDF: title page + 5 plots.

Usage
-----
    python validation_report.py
    python validation_report.py --input-json results.json --output-pdf report.pdf
    python validation_report.py --profile-name "AASM v3 (recommended)"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
except ImportError:
    print("matplotlib is required. Install with: pip install psgscoring[plot]",
          file=sys.stderr)
    sys.exit(1)


# ── AASM severity tiers ────────────────────────────────────────────────────
SEVERITY_BOUNDS = [(0, 5, "Normal"), (5, 15, "Mild"),
                   (15, 30, "Moderate"), (30, 200, "Severe")]
SEVERITY_NAMES = ["Normal", "Mild", "Moderate", "Severe"]
SEVERITY_COLORS = {
    "Normal":   "#a8e6a3",
    "Mild":     "#fff2a8",
    "Moderate": "#ffc987",
    "Severe":   "#ff8a80",
}
ALGO_COLOR = "#1f77b4"
REF_COLOR  = "#ff7f0e"
GRADE_COLORS = {"A": "#2ca02c", "B": "#ff9f1c", "C": "#d62728"}


def severity(ahi: float) -> str:
    for lo, hi, name in SEVERITY_BOUNDS:
        if lo <= ahi < hi:
            return name
    return "Severe"


def add_severity_bands(ax, axis: str = "x", alpha: float = 0.15,
                       max_val: float | None = None) -> None:
    """Shade severity zones along the given axis. max_val caps the top band
    so axhspan/axvspan does not blow up the auto-scaled axis range."""
    bounds = SEVERITY_BOUNDS
    if max_val is not None:
        bounds = [(lo, min(hi, max_val), n) for lo, hi, n in bounds if lo < max_val]
    for lo, hi, name in bounds:
        if axis == "x":
            ax.axvspan(lo, hi, alpha=alpha, color=SEVERITY_COLORS[name], zorder=0)
        else:
            ax.axhspan(lo, hi, alpha=alpha, color=SEVERITY_COLORS[name], zorder=0)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_scatter(ax, recordings, ref, algo) -> None:
    ax.set_title("Algorithm AHI vs scorer median (per recording)", fontweight="bold")
    add_severity_bands(ax, "x")
    add_severity_bands(ax, "y")
    lim = max(max(ref), max(algo)) * 1.15 + 5
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="identity")
    for tier in (5, 15, 30):
        ax.axvline(tier, color="gray", lw=0.5, alpha=0.4)
        ax.axhline(tier, color="gray", lw=0.5, alpha=0.4)
    ax.scatter(ref, algo, s=80, color=ALGO_COLOR, edgecolor="black",
               linewidth=0.5, zorder=3)
    for rec, x, y in zip(recordings, ref, algo):
        ax.annotate(rec, (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Scorer median AHI (events/h)")
    ax.set_ylabel("Algorithm AHI (events/h)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)


def plot_bland_altman(ax, recordings, ref, algo, bias, loa_low, loa_high) -> None:
    means = (np.array(ref) + np.array(algo)) / 2
    diffs = np.array(algo) - np.array(ref)
    ax.set_title("Bland-Altman: algorithm − scorer median", fontweight="bold")
    ax.axhline(bias, color=ALGO_COLOR, lw=2, label=f"bias = {bias:+.2f}")
    ax.axhline(loa_low,  color="gray", lw=1.2, ls="--",
               label=f"95% LoA = [{loa_low:+.2f}, {loa_high:+.2f}]")
    ax.axhline(loa_high, color="gray", lw=1.2, ls="--")
    ax.axhspan(loa_low, loa_high, alpha=0.08, color=ALGO_COLOR)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.scatter(means, diffs, s=80, color=ALGO_COLOR, edgecolor="black",
               linewidth=0.5, zorder=3)
    for rec, x, y in zip(recordings, means, diffs):
        ax.annotate(rec, (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean AHI ((algorithm + scorer) / 2)")
    ax.set_ylabel("Difference (algorithm − scorer)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    n = len(recordings)
    ax.text(0.02, 0.02,
            f"n = {n}. With small n the 95% LoA bands have wide CI;\n"
            f"interpret bias as the primary point estimate.",
            transform=ax.transAxes, fontsize=8, style="italic",
            color="#555", verticalalignment="bottom")


def plot_confusion(ax, ref, algo, kappa) -> None:
    ax.set_title("Severity concordance", fontweight="bold")
    matrix = np.zeros((4, 4), dtype=int)
    for r, a in zip(ref, algo):
        matrix[SEVERITY_NAMES.index(severity(a)),
               SEVERITY_NAMES.index(severity(r))] += 1
    n_total = matrix.sum()
    n_diag  = int(np.trace(matrix))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    for i in range(4):
        for j in range(4):
            v = matrix[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v >= matrix.max() * 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(SEVERITY_NAMES)
    ax.set_yticks(range(4)); ax.set_yticklabels(SEVERITY_NAMES)
    ax.set_xlabel("Scorer median severity")
    ax.set_ylabel("Algorithm severity")
    ax.set_title(
        f"Severity concordance — "
        f"{n_diag}/{n_total} ({100*n_diag/n_total:.0f}%), weighted κ = {kappa:.2f}",
        fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recordings")


def plot_per_recording(ax, recs, ref_med, ref_lo, ref_hi, algo_std) -> None:
    ax.set_title("Per-recording: scorer range vs algorithm", fontweight="bold")
    ymax = max(max(ref_hi), max(algo_std)) * 1.20 + 5
    add_severity_bands(ax, "y", max_val=ymax)
    ax.set_ylim(0, ymax)
    x = np.arange(len(recs))
    for i, (lo, hi, med) in enumerate(zip(ref_lo, ref_hi, ref_med)):
        ax.vlines(i, lo, hi, color=REF_COLOR, lw=8, alpha=0.5,
                  label="scorer min-max" if i == 0 else None)
        ax.scatter(i, med, color=REF_COLOR, s=60, zorder=4, edgecolor="black",
                   linewidth=0.5, label="scorer median" if i == 0 else None)
    ax.scatter(x, algo_std, color=ALGO_COLOR, s=120, marker="D", zorder=5,
               edgecolor="black", linewidth=0.5, label="algorithm")
    ax.set_xticks(x); ax.set_xticklabels(recs)
    ax.set_xlabel("Recording")
    ax.set_ylabel("AHI (events/h)")
    ax.set_xlim(-0.5, len(recs) - 0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2)


def plot_profile_sweep(ax, recs, strict, standard, sensitive, ref_med, grades) -> None:
    ax.set_title("AHI confidence interval (strict / standard / sensitive)",
                 fontweight="bold")
    candidates = [v for v in strict + standard + sensitive + ref_med if v is not None]
    ymax = max(candidates) * 1.20 + 5
    add_severity_bands(ax, "y", max_val=ymax)
    ax.set_ylim(0, ymax)
    x = np.arange(len(recs))
    for i, (s, m, p, grade) in enumerate(zip(strict, standard, sensitive, grades)):
        if s is not None and p is not None:
            color = GRADE_COLORS.get(grade, "gray")
            ax.vlines(i, s, p, color=color, lw=10, alpha=0.55,
                      label=f"grade {grade}" if grade and grade not in
                            [g for g in grades[:i]] else None)
        ax.scatter(i, m, color=ALGO_COLOR, s=110, marker="D", zorder=5,
                   edgecolor="black", linewidth=0.5,
                   label="algorithm (standard)" if i == 0 else None)
        if ref_med[i] is not None:
            ax.scatter(i, ref_med[i], color=REF_COLOR, s=70, zorder=4,
                       edgecolor="black", linewidth=0.5,
                       label="scorer median" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(recs)
    ax.set_xlabel("Recording")
    ax.set_ylabel("AHI (events/h)")
    ax.set_xlim(-0.5, len(recs) - 0.5)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2)


# ── Title page ─────────────────────────────────────────────────────────────

def render_title_page(pdf, data, meta) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.92, "PSG-IPA validation report", ha="center",
             fontsize=22, fontweight="bold")
    fig.text(0.5, 0.88, "psgscoring vs human scorer consensus",
             ha="center", fontsize=13, color="#555")

    info = [
        ("Generated",        meta["generated"]),
        ("Profile",          meta["profile_name"]),
        ("Input JSON",       meta["input_json"]),
        ("Recordings",       ", ".join(data["per_recording"].keys())),
    ]
    y = 0.78
    for k, v in info:
        fig.text(0.12, y, k, fontsize=11, fontweight="bold")
        fig.text(0.32, y, str(v), fontsize=11)
        y -= 0.03

    agg = data["aggregate"]
    fig.text(0.12, 0.62, "Aggregate metrics", fontsize=14, fontweight="bold")
    rows = [
        ("Bias",            f"{agg['bias']:+.2f} events/h"),
        ("MAE",             f"{agg['mae']:.2f} events/h"),
        ("95% LoA",         f"[{agg['loa_low']:+.2f}, {agg['loa_high']:+.2f}] events/h"),
        ("Pearson r",       f"{agg['pearson_r']:.3f}"),
        ("Weighted κ",      f"{agg['weighted_kappa']:.3f}"),
    ]
    if "sn3_event_level" in data:
        ev = data["sn3_event_level"]
        rows += [
            ("SN3 event-level F1", f"{ev['f1']:.3f}"),
            ("SN3 TP / FP / FN",   f"{ev['tp']} / {ev['fp']} / {ev['fn']}"),
            ("SN3 mean Δt",        f"{ev['dt_s']:.2f} s"),
        ]
    y = 0.57
    for k, v in rows:
        fig.text(0.14, y, k, fontsize=11)
        fig.text(0.42, y, v, fontsize=11, family="monospace")
        y -= 0.028

    fig.text(0.5, 0.10,
             "Research software — not a medical device. "
             "Outputs require physician review before clinical action.",
             ha="center", fontsize=9, style="italic", color="#777",
             wrap=True)
    fig.text(0.5, 0.07, "psgscoring · github.com/bartromb/psgscoring",
             ha="center", fontsize=9, color="#777")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-json", type=Path,
                   default=Path("/tmp/validation_results.json"))
    p.add_argument("--output-pdf", type=Path, default=Path("validation_report.pdf"))
    p.add_argument("--profile-name", default="standard")
    args = p.parse_args()

    if not args.input_json.exists():
        print(f"Input JSON not found: {args.input_json}", file=sys.stderr)
        print("Run validate_psgipa.py first.", file=sys.stderr)
        return 1

    data = json.loads(args.input_json.read_text())

    recs        = list(data["per_recording"].keys())
    per         = [data["per_recording"][r] for r in recs]
    ref_med     = [d["ref_median"] for d in per]
    ref_lo      = [d["ref_range"][0] for d in per]
    ref_hi      = [d["ref_range"][1] for d in per]
    algo_std    = [d["algo_standard"] for d in per]
    algo_strict = [d["algo_strict"] for d in per]
    algo_sens   = [d["algo_sensitive"] for d in per]
    grades      = [d.get("robustness_grade") for d in per]

    meta = {
        "generated":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "profile_name": args.profile_name,
        "input_json":   str(args.input_json),
    }

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    with PdfPages(args.output_pdf) as pdf:
        render_title_page(pdf, data, meta)

        for plot_fn in [
            lambda a: plot_scatter(a, recs, ref_med, algo_std),
            lambda a: plot_bland_altman(
                a, recs, ref_med, algo_std,
                data["aggregate"]["bias"],
                data["aggregate"]["loa_low"],
                data["aggregate"]["loa_high"]),
            lambda a: plot_confusion(
                a, ref_med, algo_std, data["aggregate"]["weighted_kappa"]),
            lambda a: plot_per_recording(
                a, recs, ref_med, ref_lo, ref_hi, algo_std),
            lambda a: plot_profile_sweep(
                a, recs, algo_strict, algo_std, algo_sens, ref_med, grades),
        ]:
            fig, ax = plt.subplots(figsize=(8.5, 7))
            plot_fn(ax)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {args.output_pdf} ({args.output_pdf.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
