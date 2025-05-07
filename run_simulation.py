import os
import random
import time
import csv
from statistics import mean

from lt.encode import encoder as lt_encoder
from lt.decode import LtDecoder, block_from_bytes


input_files = {
    "32KB": "input_32kb.txt",
    "64KB": "input_64kb.txt",
    "128KB": "input_128kb.txt",
    "256KB": "input_256kb.txt",
    "512KB": "input_512kb.txt",
    "1MB": "input_1mb.txt"
}

block_sizes = [8, 16, 64, 128]
distances = [0, 50, 100, 150, 200, 250]
trials_per_setting = 100
output_csv = "full_results.csv"


# Loss rate according to distance
def loss_rate(distance, max_range=300):
    if distance >= max_range:
        return 0.98
    return min(0.1 + 0.002 * distance, 0.9)


def run_trial(file_path, blocksize, distance):
    with open(file_path, 'rb') as f:
        symbol_generator = lt_encoder(f, blocksize)
        decoder = LtDecoder()
        symbols_generated = 0
        symbols_received = 0
        loss_prob = loss_rate(distance)
        start_time = time.time()

        while not decoder.is_done():
            encoded_symbol = next(symbol_generator)
            symbols_generated += 1

            if random.random() > loss_prob:
                symbols_received += 1
                try:
                    parsed_block = block_from_bytes(encoded_symbol)
                    decoder.consume_block(parsed_block)
                except Exception:
                    continue  # Skip corrupt or partial symbol

        elapsed = time.time() - start_time
        return symbols_generated, elapsed



results = [("file_size", "block_size", "distance", "loss_rate", "avg_symbols_needed", "effective_rate")]

for file_label, file_path in input_files.items():
    file_size_bytes = os.path.getsize(file_path)

    for blocksize in block_sizes:
        K = file_size_bytes // blocksize + (1 if file_size_bytes % blocksize else 0)

        print(f"\n BLOCK SIZE: {blocksize} bytes")

        for distance in distances:
            trial_symbols = []
            trial_times = []
        
            print(f"DISTANCE: {distance}m")
            for i in range(trials_per_setting):
                print(f"\n TRIAL NO.{i + 1}/{trials_per_setting}")
                symbols, elapsed = run_trial(file_path, blocksize, distance)
                trial_symbols.append(symbols)
                trial_times.append(elapsed)

            avg_symbols = round(mean(trial_symbols), 2)
            avg_time = round(mean(trial_times), 2)
            loss = round(loss_rate(distance), 3)
            eff_rate = round(K / avg_symbols, 4)

            results.append((file_label, blocksize, distance, loss, avg_symbols, eff_rate, avg_time))
            print(f"\n {file_label} | Block {blocksize} | Dist {distance}m → Avg sym: {avg_symbols}, EffRate: {eff_rate}, Time: {avg_time}s")


# save to csv file
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print(f"\n Results saved to: {output_csv}")

