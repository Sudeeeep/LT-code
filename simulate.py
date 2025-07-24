import cv2
import os
import time
import csv
import random
import numpy as np
import io
from lt.encode import encoder as lt_encoder
from lt.decode import LtDecoder, block_from_bytes
import tkinter as tk
from collections import defaultdict
from encode_video import encode_multiple_res_video


input_video_path = "input_video_10mb.mp4"
output_video_path = "output_adaptive.mp4"
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
grid_size = 5
cell_size = 100
uav_altitude = 100

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

for i in range(grid_size):
    x = i * cell_size
    canvas.create_line(x, 0, x, trace_area[1], fill="lightgray")
    canvas.create_line(0, x, trace_area[0], x, fill="lightgray")

def compute_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return ((x1 - x2)**2 + (y1 - y2)**2 + (uav_altitude)**2) ** 0.5

def loss_rate(distance, max_range=500):
    if distance >= max_range:
        return 0.98
    return min(0.1 + 0.002 * distance, 0.9)

def generate_manhattan_grid_trace(start_position, grid_size, cell_size, time_step, speed_mps):
    x, y = int(start_position[0] // cell_size), int(start_position[1] // cell_size)
    current_index = (x, y)

    def select_random_direction(x, y):
        directions = []
        if x < (grid_size - 1): 
            directions.append((1, 0))
        if x > 0: 
            directions.append((-1, 0))
        if y < (grid_size - 1): 
            directions.append((0, 1))
        if y > 0: 
            directions.append((0, -1))
        return random.choice(directions)

    direction = select_random_direction(*current_index)

    while True:
        next_index = (current_index[0] + direction[0], current_index[1] + direction[1])
        start_pos = (current_index[0] * cell_size, current_index[1] * cell_size)
        end_pos = (next_index[0] * cell_size, next_index[1] * cell_size)

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2) ** 0.5
        travel_time = distance / speed_mps
        steps = max(1, int(travel_time / time_step))

        for step in range(steps):
            alpha = step / steps
            x = start_pos[0] + alpha * dx
            y = start_pos[1] + alpha * dy
            yield (x, y)

        current_index = next_index
        direction = select_random_direction(*current_index)


def simulate_frame_transmission(frame_data, mobility_gen, symbols_per_step):
    symbol_generator = lt_encoder(io.BytesIO(frame_data), block_size)
    decoder = LtDecoder()
    symbols_sent = 0
    symbols_received = 0
    distances = []


    K = len(frame_data) // block_size + 1

    start_time = time.time()

    while not decoder.is_done():
        position = next(mobility_gen)
        print("current position:", position)
        update_uav_on_canvas(*position)
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
        return decoder.bytes_dump(), symbols_sent, latency, avg_distance, effective_rate
    else:
        return None, symbols_sent, latency, avg_distance, effective_rate

def run_video_simulation(range_data):
    resolutions = ['144p', '360p', '480p', '720p']
    resolution_caps = {}
    for res in resolutions:
        resolution_caps[res] = cv2.VideoCapture(os.path.join('res_videos', f'video_{res}.mp4'))

    all_opened = True
    for cap in resolution_caps.values():
        if not cap.isOpened():
            all_opened = False
            break
    if not all_opened:
        print("video files couldn't be opened.")
        return
    
    fps = resolution_caps['720p'].get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    prev_output_size = None
    frame_count = 0
    x = random.randint(0, grid_size - 1)
    y = random.randint(0, grid_size - 1)
    initial_position = (x * cell_size, y * cell_size)
    mobility_gen = generate_manhattan_grid_trace(initial_position, grid_size, cell_size, simulation_time_step, uav_speed)
    quality_index = 0

    while True:
        frames = {}
        for res, cap in resolution_caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, f = cap.read()
            if not ret:
                frames[res] = None
            else:
                frames[res] = f

        all_ended = True
        for f in frames.values():
            if f is not None:
                all_ended = False
                break
        if all_ended:
            break

        selected_res = resolutions[quality_index]
        frame = frames[selected_res]

        if frame is None:
            print(f"Frame {frame_count} not available in resolution {selected_res}")
            frame_count += 1
            continue

        _, compressed = cv2.imencode('.jpg', frame)
        frame_bytes = compressed.tobytes()
        bits_per_sec = bitrate_Mbps * 1_000_000
        bits_per_symbol = block_size * 8
        symbols_per_sec = bits_per_sec / bits_per_symbol
        symbols_per_step = max(1, int(symbols_per_sec * simulation_time_step))

        decoded_data, symbols, latency, avg_distance, eff_rate = simulate_frame_transmission(frame_bytes, mobility_gen, symbols_per_step)

        latency = max(latency, 1e-6)
        throughput = (len(frame_bytes) * 8) / (latency * 1_000_000)

        range_mean = round((avg_distance // 10) * 10 + 5)

        range_data[range_mean]["symbols_needed"].append(symbols)
        range_data[range_mean]["effective_rate"].append(eff_rate)
        range_data[range_mean]["resolution"].append(selected_res)

        if latency > 0.01 and throughput < 1000:
            range_data[range_mean]["latency_sec"].append(latency)
            range_data[range_mean]["throughput_Mbps"].append(throughput)


        if decoded_data:
            decoded_img = cv2.imdecode(np.frombuffer(decoded_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded_img is not None:
                h, w = decoded_img.shape[:2]
                if out is None or (w, h) != prev_output_size:
                    if out is not None:
                        out.release()
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                    prev_output_size = (w, h)

                cv2.putText(decoded_img, selected_res, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                out.write(decoded_img)
                cv2.imwrite(os.path.join(frame_dump_dir, f"frame_{frame_count}.jpg"), decoded_img)
            else:
                print(f"Skipped frame {frame_count}: decode frame size issue")
        else:
            print(f"Skipped frame {frame_count}: LT decoding failed entirely")

        if latency > 0.8 and quality_index > 0:
            quality_index -= 1
        elif latency < 0.3 and quality_index < len(resolutions) - 1:
            quality_index += 1

        print(f"Frame {frame_count} | {selected_res} → Symbols: {symbols}, Latency: {latency:.6f}s, "
              f"Throughput: {throughput:.2f} Mbps, Avg Distance: {avg_distance}, Effective Rate: {eff_rate}")
        frame_count += 1

    for cap in resolution_caps.values():
        cap.release()
    out.release()

def run_multiple_trials():
    range_data = defaultdict(lambda: defaultdict(list))

    for trial in range(trials):
        print(f"\n----- TRIAL {trial + 1} -----")
        run_video_simulation(range_data)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["range_mean", "symbols_needed", "latency_sec", "throughput_Mbps", "effective_rate", "resolution"])
        for range_mean in sorted(range_data.keys()):
            row = [
                range_mean,
                round(np.mean(range_data[range_mean]["symbols_needed"]), 2),
                round(np.mean(range_data[range_mean]["latency_sec"]), 6),
                round(np.mean(range_data[range_mean]["throughput_Mbps"]), 2),
                round(np.mean(range_data[range_mean]["effective_rate"]), 4),
                max(set(range_data[range_mean]["resolution"]), key=range_data[range_mean]["resolution"].count)
            ]
            writer.writerow(row)
            print(f"Range {range_mean}m → Samples: {len(range_data[range_mean]['latency_sec'])}")

    print(f"\nAll {trials} trials complete. Averaged metrics saved to {output_csv}")


if __name__ == "__main__":
    encode_multiple_res_video(input_video_path, "res_videos/")
    run_multiple_trials()
    window.mainloop()
