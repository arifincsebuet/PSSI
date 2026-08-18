"""
PSSI Experimental Results — Publication-Quality Figures
========================================================
Generates all 9 figures for Section 7 of the PSSI paper.

Usage:
    pip install matplotlib numpy
    python pssi_plot_results.py

Output (saved to ./figures/):
    fig1_latency_comparison.png
    fig2_memory_comparison.png
    fig3_network_payload.png
    fig4_accuracy_per_dataset.png
    fig5_privacy_leakage.png
    fig6_eta_sweep.png
    fig7_bloom_size_ablation.png
    fig8_lambda_ablation.png
    fig9_scalability.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#E5E5E5",
    "grid.linewidth":     0.8,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#CCCCCC",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "pssi":   "#1D9E75",   # teal  — PSSI
    "es":     "#378ADD",   # blue  — Elasticsearch
    "lucene": "#7F77DD",   # purple — Lucene
    "dense":  "#F5A623",   # amber  — Dense / ANCE
    "bf":     "#D85A30",   # coral  — Plain BF
    "red":    "#E24B4A",
    "gray":   "#888780",
}

HATCH = {
    "pssi":   "",
    "es":     "///",
    "lucene": "...",
    "dense":  "xxx",
    "bf":     "\\\\\\",
}

BAR_EDGE = "white"
BAR_LW   = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: annotate bar tops
# ═══════════════════════════════════════════════════════════════════════════════
def annotate_bars(ax, rects, fmt="{:.0f}", color="black", offset=3):
    for r in rects:
        h = r.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(r.get_x() + r.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color=color,
        )


def save(name):
    path = f"figures/{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Query Latency Comparison (grouped bar: avg / P95 / P99)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_latency():
    systems = ["Elasticsearch", "Lucene", "Dense\n(ANCE)", "Plain BF\n(η=0)", "PSSI\n(ours)"]
    avg = [334, 348, 407, 218, 187]
    p95 = [578, 641, 714, 391, 334]
    p99 = [892, 974, 1063, 601, 498]

    x      = np.arange(len(systems))
    width  = 0.26
    colors = [C["es"], C["lucene"], C["dense"], C["bf"], C["pssi"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_avg = ax.bar(x - width, avg, width, label="Avg latency",
                      color=colors, edgecolor=BAR_EDGE, linewidth=BAR_LW)
    bars_p95 = ax.bar(x,         p95, width, label="P95 latency",
                      color=colors, edgecolor=BAR_EDGE, linewidth=BAR_LW, alpha=0.70)
    bars_p99 = ax.bar(x + width, p99, width, label="P99 latency",
                      color=colors, edgecolor=BAR_EDGE, linewidth=BAR_LW, alpha=0.42)

    for rects, data in [(bars_avg, avg), (bars_p95, p95), (bars_p99, p99)]:
        for rect, val in zip(rects, data):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 12,
                    str(val), ha="center", va="bottom", fontsize=8)

    # Reduction annotation on PSSI avg bar
    ax.annotate("−44%\nvs ES", xy=(x[-1] - width, avg[-1]),
                xytext=(x[-1] - width - 0.55, avg[-1] + 120),
                fontsize=8.5, color=C["pssi"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["pssi"], lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fig 1.  Query Latency Comparison (Avg / P95 / P99)")
    ax.set_ylim(0, 1200)

    # Custom legend: shading encodes metric group
    from matplotlib.patches import Patch
    leg_handles = [
        Patch(facecolor="#AAAAAA", label="Avg latency (full opacity)"),
        Patch(facecolor="#AAAAAA", alpha=0.70, label="P95 latency (70% opacity)"),
        Patch(facecolor="#AAAAAA", alpha=0.42, label="P99 latency (42% opacity)"),
    ]
    color_handles = [Patch(facecolor=c, label=s.replace("\n", " "))
                     for s, c in zip(systems, colors)]
    ax.legend(handles=leg_handles + color_handles, ncol=2,
              loc="upper left", fontsize=9)

    fig.tight_layout()
    save("fig1_latency_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Memory Footprint
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_memory():
    systems = ["Elasticsearch\n(BM25)", "Lucene\n(BM25)", "Dense\n(ANCE)",
               "Plain BF\n(η=0)", "PSSI\n(ours)"]
    memory  = [2.63, 7.5, 11.6, 2.1, 1.50]
    colors  = [C["es"], C["lucene"], C["dense"], C["bf"], C["pssi"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(systems, memory, color=colors,
                  edgecolor=BAR_EDGE, linewidth=BAR_LW, width=0.55)

    for bar, val in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{val:g} GB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Percentage labels relative to ES
    for bar, val in zip(bars, memory):
        if val != memory[0]:
            pct = (val - memory[0]) / memory[0] * 100
            label = f"{pct:+.0f}%"
            color = C["pssi"] if pct < -50 else C["red"] if pct > 0 else C["gray"]
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    label, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_ylabel("Peak Server Memory (GB)")
    ax.set_title("Fig 2.  Server-Side Memory Consumption")
    ax.set_ylim(0, 13)
    ax.axhline(memory[0], linestyle="--", color=C["es"], alpha=0.4, linewidth=1)
    ax.text(4.4, memory[0] + 0.2, "ES baseline", fontsize=8, color=C["es"])

    fig.tight_layout()
    save("fig2_memory_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Network Payload per Query
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_network():
    systems = ["Elasticsearch", "Lucene", "Dense\n(ANCE)*", "Plain BF\n(η=0)", "PSSI\n(ours)"]
    payload = [38.4, 36.1, 5.2, 11.8, 11.2]
    colors  = [C["es"], C["lucene"], C["dense"], C["bf"], C["pssi"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(systems, payload, color=colors,
                  edgecolor=BAR_EDGE, linewidth=BAR_LW, width=0.55)

    for bar, val in zip(bars, payload):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val} KB", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Avg Payload per Query (KB)")
    ax.set_title("Fig 3.  Network Payload per Query (Client → Cloud)")
    ax.set_ylim(0, 50)

    ax.annotate("* Dense transmits\na float32 vector\n(privacy risk)", xy=(2, 5.2),
                xytext=(2.6, 22), fontsize=8.5, color=C["dense"],
                ha="center",
                arrowprops=dict(arrowstyle="->", color=C["dense"], lw=1.0))

    fig.tight_layout()
    save("fig3_network_payload")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Retrieval Accuracy per Dataset (grouped: P@10 / F1@10 / nDCG@10)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_accuracy():
    datasets = ["NFCorpus", "SciFact", "FiQA-2018"]
    data = {
        "Elasticsearch": {
            "P@10":   [0.541, 0.576, 0.477],
            "F1@10":  [0.503, 0.531, 0.432],
            "nDCG@10":[0.617, 0.648, 0.541],
        },
        "Dense (ANCE)": {
            "P@10":   [0.612, 0.634, 0.557],
            "F1@10":  [0.571, 0.589, 0.512],
            "nDCG@10":[0.694, 0.721, 0.627],
        },
        "Plain BF": {
            "P@10":   [0.489, 0.498, 0.447],
            "F1@10":  [0.451, 0.462, 0.409],
            "nDCG@10":[0.558, 0.573, 0.511],
        },
        "PSSI (ours)": {
            "P@10":   [0.504, 0.521, 0.455],
            "F1@10":  [0.469, 0.484, 0.424],
            "nDCG@10":[0.578, 0.601, 0.531],
        },
    }
    sys_colors = [C["es"], C["dense"], C["bf"], C["pssi"]]
    metrics    = ["P@10", "F1@10", "nDCG@10"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for col, metric in enumerate(metrics):
        ax = axes[col]
        x  = np.arange(len(datasets))
        n  = len(data)
        w  = 0.18
        offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

        for i, (sys_name, vals) in enumerate(data.items()):
            bars = ax.bar(x + offsets[i], vals[metric], w,
                          color=sys_colors[i], label=sys_name,
                          edgecolor=BAR_EDGE, linewidth=BAR_LW)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(f"{metric}")
        ax.set_ylim(0.30, 0.80)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if col == 0:
            ax.set_ylabel("Score")

    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Fig 4.  Retrieval Accuracy by Dataset and System (top-10)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig4_accuracy_per_dataset")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Privacy Leakage Comparison (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_leakage():
    systems = [
        "Elasticsearch (BM25)",
        "Lucene (BM25)",
        "Dense retrieval (ANCE)",
        "Plain BF index (η=0)",
        "PSSI — η=0.02",
        "PSSI — η=0.05 (default)",
        "PSSI — η=0.10",
    ]
    leakage = [0.82, 0.82, 0.65, 0.35, 0.22, 0.08, 0.06]
    colors  = [C["es"], C["lucene"], C["dense"], C["bf"],
               "#9FE1CB", C["pssi"], "#085041"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(systems))
    bars = ax.barh(y, leakage, color=colors, edgecolor=BAR_EDGE,
                   linewidth=BAR_LW, height=0.55)

    for bar, val in zip(bars, leakage):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(systems)
    ax.set_xlabel("Token Reconstruction Probability (L)")
    ax.set_title("Fig 5.  Privacy Leakage Probability Across Systems")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.10, linestyle="--", color=C["pssi"], linewidth=1.2, alpha=0.7)
    ax.text(0.105, -0.7, "L = 0.10\nthreshold", fontsize=8, color=C["pssi"])

    # Danger / safe zones
    ax.axvspan(0, 0.10,  alpha=0.06, color=C["pssi"])
    ax.axvspan(0.10, 0.50, alpha=0.04, color=C["dense"])
    ax.axvspan(0.50, 1.05, alpha=0.05, color=C["red"])

    ax.text(0.005, 6.55, "Safe zone", fontsize=8, color=C["pssi"], alpha=0.8)
    ax.text(0.52,  6.55, "High risk", fontsize=8, color=C["red"],   alpha=0.8)

    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save("fig5_privacy_leakage")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — η Sweep: Privacy-Utility Tradeoff (dual-axis)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_eta_sweep():
    eta     = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    f1      = [0.478, 0.475, 0.469, 0.451, 0.429, 0.401, 0.348, 0.261]
    leakage = [0.35,  0.22,  0.08,  0.06,  0.05,  0.04,  0.04,  0.04]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    line1, = ax1.plot(eta, f1, "o-", color=C["pssi"],
                      linewidth=2.2, markersize=7, label="F1@10 (left axis)", zorder=3)
    ax1.fill_between(eta, f1, alpha=0.12, color=C["pssi"])

    line2, = ax2.plot(eta, leakage, "s--", color=C["red"],
                      linewidth=2.0, markersize=7, label="Leakage L (right axis)", zorder=3)
    ax2.fill_between(eta, leakage, alpha=0.08, color=C["red"])

    # Mark default η = 0.05
    ax1.axvline(0.05, linestyle=":", color="#444444", linewidth=1.3, alpha=0.7)
    ax1.text(0.052, 0.245, "default\nη = 0.05", fontsize=8.5, color="#444444")

    # Annotations at default
    ax1.annotate(f"F1 = {f1[2]:.3f}", xy=(0.05, f1[2]),
                 xytext=(0.10, f1[2] + 0.015),
                 fontsize=9, color=C["pssi"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["pssi"], lw=1.0))
    ax2.annotate(f"L = {leakage[2]:.2f}", xy=(0.05, leakage[2]),
                 xytext=(0.12, leakage[2] + 0.04),
                 fontsize=9, color=C["red"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0))

    ax1.set_xlabel("Noise parameter η (bit-flip probability)")
    ax1.set_ylabel("F1@10", color=C["pssi"])
    ax2.set_ylabel("Leakage probability L", color=C["red"])
    ax1.tick_params(axis="y", colors=C["pssi"])
    ax2.tick_params(axis="y", colors=C["red"])
    ax1.set_ylim(0.22, 0.52)
    ax2.set_ylim(0.00, 0.42)
    ax1.set_title("Fig 6.  Privacy-Utility Tradeoff: Effect of Noise Parameter η")
    ax1.grid(True, alpha=0.4)
    ax2.grid(False)

    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    save("fig6_eta_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Bloom Filter Size Ablation (multi-metric)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_bloom_size():
    m_vals   = [512, 1024, 2048, 4096]
    f1       = [0.431, 0.469, 0.481, 0.484]
    fpr      = [5.64,  0.86,  0.12,  0.02]   # k-hash FPR (%) = occupancy^k, k=4
    payload  = [14.2,  11.2,  18.7,  34.1]   # KB
    memory   = [1.2,   1.5,   2.9,   5.8]    # GB

    x     = np.arange(len(m_vals))
    x_lbl = [str(v) for v in m_vals]
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Panel A: F1 vs m
    ax = axes[0]
    bars = ax.bar(x, f1, color=C["pssi"], edgecolor=BAR_EDGE,
                  linewidth=BAR_LW, width=0.5)
    bars[1].set_edgecolor(C["pssi"])
    bars[1].set_linewidth(2.5)      # highlight default m=1024
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(x_lbl)
    ax.set_xlabel("Bloom filter size m (bits)")
    ax.set_ylabel("F1@10")
    ax.set_title("A.  Retrieval F1@10")
    ax.set_ylim(0.40, 0.52)
    ax.axvline(1, linestyle=":", color="#888888", linewidth=1.2)
    ax.text(1.1, 0.408, "default", fontsize=8, color="#888888")

    # Panel B: False positive rate vs m
    ax = axes[1]
    ax.plot(x, fpr, "o-", color=C["red"], linewidth=2.2, markersize=8)
    ax.set_yscale("log")
    for xi, val in zip(x, fpr):
        ax.text(xi, val * 1.35, f"{val:g}%", ha="center", va="bottom", fontsize=9)
    ax.axvline(1, linestyle=":", color="#888888", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(x_lbl)
    ax.set_xlabel("Bloom filter size m (bits)")
    ax.set_ylabel("False Positive Rate (%, log scale)")
    ax.set_title("B.  Bloom Filter False Positive Rate")
    ax.set_ylim(0.008, 30)

    # Panel C: Payload vs m (dual bar: payload + memory)
    ax   = axes[2]
    ax2  = ax.twinx()
    b1   = ax.bar(x - 0.12, payload, 0.24, color=C["es"],   label="Payload (KB)", edgecolor=BAR_EDGE)
    b2   = ax2.bar(x + 0.12, memory, 0.24, color=C["dense"], label="Memory (GB)",  edgecolor=BAR_EDGE)
    ax.set_xticks(x); ax.set_xticklabels(x_lbl)
    ax.set_xlabel("Bloom filter size m (bits)")
    ax.set_ylabel("Payload (KB)", color=C["es"])
    ax2.set_ylabel("Memory (GB)", color=C["dense"])
    ax.tick_params(axis="y", colors=C["es"])
    ax2.tick_params(axis="y", colors=C["dense"])
    ax.set_title("C.  Payload & Memory Cost")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.suptitle("Fig 7.  Ablation: Bloom Filter Size m", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig7_bloom_size_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — λ Ablation (line plot, 3 query types)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_lambda():
    lam    = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    substr = [0.401, 0.433, 0.452, 0.459, 0.461, 0.449, 0.424]
    sem    = [0.441, 0.461, 0.459, 0.458, 0.452, 0.431, 0.401]
    mixed  = [0.358, 0.406, 0.443, 0.461, 0.469, 0.461, 0.441]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lam, substr, "o-", color=C["es"],    linewidth=2.2, markersize=7,
            label="Substring queries")
    ax.plot(lam, sem,    "s-", color=C["dense"], linewidth=2.2, markersize=7,
            label="Semantic queries")
    ax.plot(lam, mixed,  "^-", color=C["pssi"],  linewidth=2.2, markersize=7,
            label="Mixed queries")

    ax.fill_between(lam, substr, alpha=0.07, color=C["es"])
    ax.fill_between(lam, sem,    alpha=0.07, color=C["dense"])
    ax.fill_between(lam, mixed,  alpha=0.07, color=C["pssi"])

    # Mark default λ = 0.5
    ax.axvline(0.5, linestyle=":", color="#555555", linewidth=1.3)
    ax.text(0.52, 0.356, "default λ = 0.5", fontsize=8.5, color="#555555")

    # Annotations at optimal points
    ax.annotate("Best for\nsubstring", xy=(0.6, max(substr)),
                xytext=(0.72, 0.475), fontsize=8.5, color=C["es"],
                arrowprops=dict(arrowstyle="->", color=C["es"], lw=1.0))
    ax.annotate("Best for\nsemantic",  xy=(0.2, sem[1]),
                xytext=(0.02, 0.475), fontsize=8.5, color=C["dense"],
                arrowprops=dict(arrowstyle="->", color=C["dense"], lw=1.0))

    ax.set_xlabel("λ  (0 = semantic-only  →  1 = substring-only)")
    ax.set_ylabel("F1@10")
    ax.set_title("Fig 8.  Ablation: Composite Weighting λ by Query Type")
    ax.set_ylim(0.34, 0.50)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(lam)
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    save("fig8_lambda_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Scalability (dual-panel: latency + memory vs corpus size)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_scalability():
    sizes   = [1_000, 10_000, 30_000, 57_638]
    labels  = ["1K", "10K", "30K", "57K"]

    pssi_lat = [43,  91,  141, 187]
    es_lat   = [102, 187, 279, 334]
    pssi_mem = [0.06, 0.21, 0.72, 1.50]
    es_mem   = [0.05, 0.48, 1.31, 2.63]   # note ES starts lower (heap overhead amortised)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Latency ────────────────────────────────────────────────
    ax1.plot(labels, pssi_lat, "o-", color=C["pssi"], linewidth=2.5,
             markersize=9, label="PSSI (ours)", zorder=3)
    ax1.plot(labels, es_lat,   "s-", color=C["es"],   linewidth=2.5,
             markersize=9, label="Elasticsearch (BM25)", zorder=3)
    ax1.fill_between(labels, pssi_lat, es_lat,
                     alpha=0.12, color=C["pssi"])

    for xi, (pv, ev) in enumerate(zip(pssi_lat, es_lat)):
        ax1.text(xi, pv - 14, str(pv), ha="center", fontsize=9,
                 color=C["pssi"], fontweight="bold")
        ax1.text(xi, ev + 6,  str(ev), ha="center", fontsize=9,
                 color=C["es"],   fontweight="bold")

    ax1.set_xlabel("Corpus size (# documents)")
    ax1.set_ylabel("Avg query latency (ms)")
    ax1.set_title("A.  Query Latency vs Corpus Size")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 420)

    # Trend annotation
    ax1.annotate("Sub-linear growth\nfor PSSI",
                 xy=("30K", 141), xytext=("10K", 290),
                 fontsize=8.5, color=C["pssi"],
                 arrowprops=dict(arrowstyle="->", color=C["pssi"], lw=1.1))

    # ── Panel B: Memory ─────────────────────────────────────────────────
    ax2.plot(labels, pssi_mem, "o-", color=C["pssi"], linewidth=2.5,
             markersize=9, label="PSSI (ours)", zorder=3)
    ax2.plot(labels, es_mem,   "s-", color=C["es"],   linewidth=2.5,
             markersize=9, label="Elasticsearch (BM25)", zorder=3)
    ax2.fill_between(labels, pssi_mem, es_mem,
                     alpha=0.10, color=C["pssi"])

    for xi, (pv, ev) in enumerate(zip(pssi_mem, es_mem)):
        ax2.text(xi, pv - 0.10, f"{pv:.2f}", ha="center", fontsize=9,
                 color=C["pssi"], fontweight="bold")
        ax2.text(xi, ev + 0.04, f"{ev:.2f}", ha="center", fontsize=9,
                 color=C["es"],   fontweight="bold")

    ax2.set_xlabel("Corpus size (# documents)")
    ax2.set_ylabel("Peak server memory (GB)")
    ax2.set_title("B.  Memory Consumption vs Corpus Size")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, 3.2)

    # Shaded saving zone
    xs = np.arange(4)
    ax2.fill_between(xs, pssi_mem, es_mem,
                     alpha=0.18, color=C["pssi"],
                     label="_nolegend_")
    ax2.text(2.6, 1.0, "Memory\nsaved", fontsize=9, color=C["pssi"],
             alpha=0.85, ha="center")

    fig.suptitle("Fig 9.  Scalability: PSSI vs Elasticsearch on FiQA-2018 Subsets",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig9_scalability")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating PSSI publication figures...\n")
    fig1_latency()
    fig2_memory()
    fig3_network()
    fig4_accuracy()
    fig5_leakage()
    fig6_eta_sweep()
    fig7_bloom_size()
    fig8_lambda()
    fig9_scalability()
    print(f"\nAll 9 figures saved to ./figures/")
    print("Replace hardcoded values with your measured results before submission.")
