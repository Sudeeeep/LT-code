
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

data = []
with open("full_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "file_size": row["file_size"],
            "block_size": int(row["block_size"]),
            "distance": int(row["distance"]),
            "avg_symbols_needed": float(row["avg_symbols_needed"]),
            "effective_rate": float(row["effective_rate"])
        })


sorted_order = ["32KB", "64KB", "128KB", "256KB", "512KB", "1MB"]
file_sizes = sorted(set(row["file_size"] for row in data), key=lambda x: sorted_order.index(x))
block_sizes = sorted(set(row["block_size"] for row in data))


grouped_data = defaultdict(lambda: defaultdict(list))
for row in data:
    grouped_data[row["file_size"]][row["block_size"]].append((row["distance"], row["avg_symbols_needed"]))

# Avg Symbols Needed vs Distance (subplots per file size)
fig, axes = plt.subplots(nrows=(len(file_sizes) + 1) // 2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for idx, file_size in enumerate(file_sizes):
    ax = axes[idx]
    block_dict = grouped_data[file_size]
    for block_size in sorted(block_dict.keys()):
        points = sorted(block_dict[block_size])
        distances = [d for d, _ in points]
        symbols = [s for _, s in points]
        ax.plot(distances, symbols, marker='o', label=f"{block_size}B")
    ax.set_title(f"File Size: {file_size}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Avg Symbols Needed")
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Avg. Symbols Needed vs Distance (per File Size)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("avg_symbols_subplots_by_file_size.png")
plt.show()

eff_data = defaultdict(lambda: defaultdict(list))
for row in data:
    eff_data[row["file_size"]][row["block_size"]].append((row["distance"], row["effective_rate"]))

# Effective Rate vs Distance (subplots per file size)
fig, axes = plt.subplots(nrows=(len(file_sizes) + 1) // 2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for idx, file_size in enumerate(file_sizes):
    ax = axes[idx]
    block_dict = eff_data[file_size]
    for block_size in sorted(block_dict.keys()):
        points = sorted(block_dict[block_size])
        distances = [d for d, _ in points]
        rates = [r for _, r in points]
        ax.plot(distances, rates, marker='o', label=f"{block_size}B")
    ax.set_title(f"File Size: {file_size}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Effective Rate")
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Effective Rate vs Distance (per File Size)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("eff_rate_subplots_by_file_size.png")
plt.show()

grouped_by_block = defaultdict(lambda: defaultdict(list))
for row in data:
    grouped_by_block[row["block_size"]][row["file_size"]].append((row["distance"], row["avg_symbols_needed"]))

#Avg Symbols Needed vs Distance (subplots per block size)
fig, axes = plt.subplots(nrows=(len(block_sizes) + 1) // 2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for idx, block_size in enumerate(block_sizes):
    ax = axes[idx]
    file_dict = grouped_by_block[block_size]
    for file_size in file_sizes:
        if file_size in file_dict:
            points = sorted(file_dict[file_size])
            distances = [d for d, _ in points]
            symbols = [s for _, s in points]
            ax.plot(distances, symbols, marker='o', label=file_size)
    ax.set_title(f"Block Size: {block_size}B")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Avg Symbols Needed")
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Avg. Symbols Needed vs Distance (per Block Size)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("avg_symbols_subplots_by_block_size.png")
plt.show()


eff_block_data = defaultdict(lambda: defaultdict(list))
for row in data:
    eff_block_data[row["block_size"]][row["file_size"]].append((row["distance"], row["effective_rate"]))

# Effective Rate vs Distance (subplots per block size)
fig, axes = plt.subplots(nrows=(len(block_sizes) + 1) // 2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for idx, block_size in enumerate(block_sizes):
    ax = axes[idx]
    file_dict = eff_block_data[block_size]
    for file_size in file_sizes:
        if file_size in file_dict:
            points = sorted(file_dict[file_size])
            distances = [d for d, _ in points]
            rates = [r for _, r in points]
            ax.plot(distances, rates, marker='o', label=file_size)
    ax.set_title(f"Block Size: {block_size}B")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Effective Rate")
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Effective Rate vs Distance (per Block Size)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("eff_rate_subplots_by_block_size.png")
plt.show()
