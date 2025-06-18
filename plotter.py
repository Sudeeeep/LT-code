import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("video_simulation_results.csv")

def plot_metric(df, x_col, y_col, ylabel, color):
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o', color=color, linewidth=2)
    plt.title(f"{ylabel} vs. Distance")
    plt.xlabel("Distance (m)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{ylabel}.png")
    plt.show()

plot_metric(df, "range_mean", "symbols_needed", "Encoded Symbols Required", "tomato")
plot_metric(df, "range_mean", "latency_sec", "Latency (seconds)", "blue")
plot_metric(df, "range_mean", "throughput_Mbps", "Throughput (Mbps)", "green")
plot_metric(df, "range_mean", "effective_rate", "Effective Rate", "purple")
