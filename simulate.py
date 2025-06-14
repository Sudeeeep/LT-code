import cv2
import os
import time
import csv
import random
import numpy as np
import io
from pymobility.models.mobility import random_waypoint

from lt.encode import encoder as lt_encoder
from lt.decode import LtDecoder, block_from_bytes
import tkinter as tk
from collections import defaultdict


input_video_path = "input_video_10mb.mp4"
output_video_path = "decoded_output.mp4"
block_size = 32  
trace_area = (500, 500)
receiver_position = (250, 250)
velocity = (1.0, 2.0)
output_csv = "video_simulation_results.csv"
frame_dump_dir = "frame_debug"
os.makedirs(frame_dump_dir, exist_ok=True)

simulation_time_step = 0.1        
uav_speed = 5               
trials = 20         
bitrate_Mbps = 6

window = tk.Tk()
window.title("UAV Simulation")
canvas = tk.Canvas(window, width=500, height=500, bg="white")
canvas.pack()
 
bx, by = receiver_position
canvas.create_rectangle(bx-10, by-10, bx+10, by+10, fill="blue", tags="base")


def update_uav_on_canvas(x, y):
    canvas.create_oval(x-1, y-1, x+1, y+1, fill="gray", outline="gray", tags="trail")

    canvas.delete("uav")
    canvas.create_oval(x-6, y-6, x+6, y+6, fill="red", tags="uav")

    canvas.update()


def compute_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

def loss_rate(distance, max_range=500):
    if distance >= max_range:
        return 0.98
    return min(0.1 + 0.002 * distance, 0.9)

def generate_interpolated_trace(model, start_position, max_steps, time_step, speed_mps):
    trace = []
    current_pos = start_position
    while len(trace) < max_steps:
        target_pos = next(model)[0]
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = compute_distance(current_pos, target_pos)
        travel_time = distance / speed_mps
        steps = max(1, int(travel_time / time_step))

        for step in range(steps):
            if len(trace) >= max_steps:
                break
            alpha = step / steps
            x = current_pos[0] + alpha * dx
            y = current_pos[1] + alpha * dy
            trace.append((x, y))

        current_pos = target_pos

    return trace


def simulate_frame_transmission(frame_data, trace, symbols_per_step):
    symbol_generator = lt_encoder(io.BytesIO(frame_data), block_size)
    decoder = LtDecoder()
    symbols_sent = 0
    symbols_received = 0
    distances = []


    K = len(frame_data) // block_size + 1

    start_time = time.time()

    for pos_index, position in enumerate(trace):
        print("current position:", position)
        update_uav_on_canvas(*position)
        if decoder.is_done():
            break
        distance = compute_distance(position, receiver_position)
        distances.append(distance)
        loss_prob = loss_rate(distance)

        for _ in range(symbols_per_step):
            if decoder.is_done():
                break
            try:
                encoded_symbol = next(symbol_generator)
                symbols_sent += 1
                if random.random() > loss_prob:
                    parsed = block_from_bytes(encoded_symbol)
                    decoder.consume_block(parsed)
                    symbols_received += 1
            except StopIteration:
                break

    latency = max(time.time() - start_time, 1e-6) 
    avg_distance = round(sum(distances) / len(distances), 2) if distances else 0

    effective_rate = round(K / symbols_sent, 4)

    if decoder.is_done():
        print("position after decoding:", position)
        return decoder.bytes_dump(), symbols_sent, latency, avg_distance, effective_rate, position
    else:
        return None, symbols_sent, latency, avg_distance, effective_rate, trace[-1]


def run_video_simulation(range_data):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    model = random_waypoint(nr_nodes=1, dimensions=trace_area, velocity=velocity)
    current_position = next(model)[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, compressed = cv2.imencode('.jpg', frame)
        frame_bytes = compressed.tobytes()
        bits_per_sec = bitrate_Mbps * 1_000_000
        bits_per_symbol = block_size * 8
        symbols_per_sec = bits_per_sec / bits_per_symbol
        symbols_per_step = max(1, int(symbols_per_sec * simulation_time_step))

        K = len(frame_bytes) // block_size + 1
        max_symbols = int(4.0 * K)
        max_trace_steps = max_symbols // symbols_per_step + 1
        trace = generate_interpolated_trace(model, current_position, max_trace_steps, simulation_time_step, uav_speed)

        decoded_data, symbols, latency, avg_distance, eff_rate, current_position = simulate_frame_transmission(
            frame_bytes, trace, symbols_per_step)

        latency = max(latency, 1e-6)
        throughput = (len(frame_bytes) * 8) / (latency * 1_000_000)

        range_mean = round((avg_distance // 10) * 10 + 5)

        range_data[range_mean]["symbols_needed"].append(symbols)
        range_data[range_mean]["effective_rate"].append(eff_rate)

        if latency > 0.01 and throughput < 1000:
            range_data[range_mean]["latency_sec"].append(latency)
            range_data[range_mean]["throughput_Mbps"].append(throughput)


        if decoded_data:
            decoded_img = cv2.imdecode(np.frombuffer(decoded_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded_img is not None and decoded_img.shape[:2] == frame.shape[:2]:
                out.write(decoded_img)
                cv2.imwrite(os.path.join(frame_dump_dir, f"frame_{frame_count}.jpg"), decoded_img)
            else:
                print(f"Skipped frame {frame_count}: image problem")
        else:
            print(f"Skipped frame {frame_count}: LT decoding failed entirely")

        print(f"Frame {frame_count} → Symbols: {symbols}, Latency: {latency:.6f}s, "
              f"Throughput: {throughput:.2f} Mbps, Avg Distance: {avg_distance}, Effective Rate: {eff_rate}")
        frame_count += 1

    cap.release()
    out.release()

def run_multiple_trials():
    range_data = defaultdict(lambda: defaultdict(list))

    for trial in range(trials):
        print(f"\n----- TRIAL {trial + 1} -----")
        run_video_simulation(range_data)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["range_mean", "symbols_needed", "latency_sec", "throughput_Mbps", "effective_rate"])
        for range_mean in sorted(range_data.keys()):
            row = [
                range_mean,
                round(np.mean(range_data[range_mean]["symbols_needed"]), 2),
                round(np.mean(range_data[range_mean]["latency_sec"]), 6),
                round(np.mean(range_data[range_mean]["throughput_Mbps"]), 2),
                round(np.mean(range_data[range_mean]["effective_rate"]), 4)
            ]
            writer.writerow(row)
            print(f"Range {range_mean}m → Samples: {len(range_data[range_mean]['latency_sec'])}")

    print(f"\nAll {trials} trials complete. Averaged metrics saved to {output_csv}")


if __name__ == "__main__":
    run_multiple_trials()
    window.mainloop()
