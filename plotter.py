import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RES_CSV = "video_simulation_results_10mb.csv"
DIAG_CSV = "link_diagnostics_10mb.csv"
OUTDIR = "plots/10mb"
os.makedirs(OUTDIR, exist_ok=True)

labels = {
    "success_rate": "Success Rate (%)",
    "latency_sec": "Latency (s)",
    "latency_all_sec": "Latency (all frames, s)",
    "throughput_Mbps": "Throughput (Mbps)",
    "effective_rate": "Effective Rate",
    "symbols_needed": "Encoded Packets Needed",
    "avg_SNR_dB": "Average SNR (dB)",
    "avg_BER": "Average BER",
    "avg_PER": "Average PER",
    "avg_PLR": "Average PLR",
    "avg_loss_prob": "Average Loss Probability",
    "loss_prob": "Loss Probability",
    "overhead": "Fountain Overhead (×K / K)",
}

def pretty_label(metric: str) -> str:
    return labels.get(metric, metric.replace("_", " ").title())

def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _sorted_unique(x):
    try:
        return sorted(list({*x}))
    except Exception:
        return list({*x})

def _label_freq(x):
    try:
        return f"{float(x):g} GHz"
    except:
        return str(x)

def normalize_columns(df):
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip().str.lower().replace({
            "uma": "UMa", "inf": "InF"
        })
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip().str.upper().replace({
            r"\s+": "_", "LOS+NLOS": "LOS_NLOS", "LOS-NLOS": "LOS_NLOS",
            "LOS ONLY": "LOS_ONLY", "LOS_ONLY": "LOS_ONLY"
        }, regex=True)
    if "freq_GHz" in df.columns:
        df["freq_GHz"] = pd.to_numeric(df["freq_GHz"], errors="coerce").round(1)

res = pd.read_csv(RES_CSV)
diag = pd.read_csv(DIAG_CSV)

normalize_columns(res)
normalize_columns(diag)
_ensure_numeric(res, [
    "freq_GHz","uav_altitude_m","range_mean",
    "symbols_needed","latency_sec", "latency_all_sec", "throughput_Mbps","effective_rate",
    "success_rate","avg_SNR_dB","avg_BER","avg_PER","avg_PLR"
])
_ensure_numeric(diag, [
    "freq_GHz","uav_altitude_m","range_mean",
    "avg_SNR_dB","avg_BER","avg_loss_prob","overhead","decoded", "latency_sec"
])

def plot_metric_vs_range(res_df, metric, ylog=False, fname_prefix=None):

    if metric not in res_df.columns:
        print(f"{metric}: not in results CSV")
        return

    scenarios = _sorted_unique(res_df["scenario"].dropna())
    freqs = _sorted_unique(res_df["freq_GHz"].dropna())
    conds = _sorted_unique(res_df["condition"].dropna())

    nrows, ncols = max(1, len(scenarios)), max(1, len(freqs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.0*nrows), squeeze=False)

    for r, scen in enumerate(scenarios):
        for c, freq in enumerate(freqs):
            ax = axes[r][c]
            sub = res_df[(res_df["scenario"] == scen) & (res_df["freq_GHz"] == freq)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for cond in conds:
                subc = sub[sub["condition"] == cond]
                if subc.empty:
                    continue
                alts = _sorted_unique(subc["uav_altitude_m"].dropna())
                for alt in alts:
                    g = subc[subc["uav_altitude_m"] == alt].sort_values("range_mean")
                    if g.empty:
                        continue
                    label = f"{cond}, {int(alt)} m"
                    ax.plot(g["range_mean"], g[metric], marker='o', linestyle='-', label=label)

            ax.set_title(f"{scen} | {_label_freq(freq)}")
            ax.set_xlabel("Range (m)")
            ax.set_ylabel(pretty_label(metric))

            if metric in ["latency_sec", "latency_all_sec"]:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

            if ylog:
                ax.set_yscale("log")
            ax.grid(True)
            ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fname = f"{metric}_vs_range.pdf" if not fname_prefix else f"{fname_prefix}_{metric}_vs_range.pdf"
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved:", path)

def plot_success_rate_from_diag(diag_df, fname_prefix="diag"):
    d = diag_df.copy()
    if "decoded" not in d.columns:
        print("no 'decoded' column")
        return

    scenarios = _sorted_unique(d["scenario"].dropna())
    freqs = _sorted_unique(d["freq_GHz"].dropna())
    conds = _sorted_unique(d["condition"].dropna())

    nrows, ncols = max(1, len(scenarios)), max(1, len(freqs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.0*nrows), squeeze=False)

    for r, scen in enumerate(scenarios):
        for c, freq in enumerate(freqs):
            ax = axes[r][c]
            sub = d[(d["scenario"] == scen) & (d["freq_GHz"] == freq)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for cond in conds:
                subc = sub[sub["condition"] == cond]
                if subc.empty:
                    continue
                alts = _sorted_unique(subc["uav_altitude_m"].dropna())
                for alt in alts:
                    g = subc[subc["uav_altitude_m"] == alt]
                    if g.empty:
                        continue
                    sr = g.groupby("range_mean")["decoded"].mean().reset_index().sort_values("range_mean")
                    label = f"{cond}, {int(alt)} m"
                    ax.plot(sr["range_mean"], 100*sr["decoded"], marker='o', linestyle='-', label=label)

            ax.set_title(f"{scen} | {_label_freq(freq)}")
            ax.set_xlabel("Range (m)")
            ax.set_ylabel(pretty_label("success_rate"))
            ax.set_ylim(0, 105)
            ax.grid(True)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("Per-frame Decode Success Rate vs Range", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"{fname_prefix}_success_rate_vs_range.pdf")
    plt.savefig(path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved:", path)

def plot_diag_metric_vs_range(diag_df, metric, ylog=False, fname_prefix="diag"):

    if metric not in diag_df.columns:
        print(f"[skip] diagnostics: {metric} not found")
        return

    scenarios = _sorted_unique(diag_df["scenario"].dropna())
    freqs = _sorted_unique(diag_df["freq_GHz"].dropna())
    conds = _sorted_unique(diag_df["condition"].dropna())

    nrows, ncols = max(1, len(scenarios)), max(1, len(freqs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.0*nrows), squeeze=False)

    for r, scen in enumerate(scenarios):
        for c, freq in enumerate(freqs):
            ax = axes[r][c]
            sub = diag_df[(diag_df["scenario"] == scen) & (diag_df["freq_GHz"] == freq)]
            if sub.empty:
                ax.set_visible(False)
                continue

            for cond in conds:
                subc = sub[sub["condition"] == cond]
                if subc.empty:
                    continue
                alts = _sorted_unique(subc["uav_altitude_m"].dropna())
                for alt in alts:
                    g = subc[subc["uav_altitude_m"] == alt].groupby("range_mean")[metric].mean().reset_index()
                    g = g.sort_values("range_mean")
                    if g.empty:
                        continue
                    label = f"{cond}, {int(alt)} m"
                    ax.plot(g["range_mean"], g[metric], marker='o', linestyle='-', label=label)

            ax.set_title(f"{scen} | {_label_freq(freq)}")
            ax.set_xlabel("Range (m)")
            ax.set_ylabel(pretty_label(metric))
            if ylog:
                ax.set_yscale("log")
            ax.grid(True)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{pretty_label(metric)} vs Range", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUTDIR, f"{fname_prefix}_{metric}_vs_range.pdf")
    plt.savefig(path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved:", path)

metrics = [
    ("success_rate", False),
    ("latency_sec", False),
    ("latency_all_sec", False),
    ("throughput_Mbps", False),
    ("effective_rate", False),
    ("symbols_needed", False),
    ("avg_SNR_dB", False),
    ("avg_BER", False),
    ("avg_PER", True),
    ("avg_PLR", True),
]
for m, ylog in metrics:
    plot_metric_vs_range(res, metric=m, ylog=ylog)


plot_success_rate_from_diag(diag, fname_prefix="diag")

plot_diag_metric_vs_range(diag, metric="avg_loss_prob", ylog=True, fname_prefix="diag")

plot_diag_metric_vs_range(diag, metric="overhead", ylog=False, fname_prefix="diag")

print("\nAll plots saved to:", OUTDIR)
