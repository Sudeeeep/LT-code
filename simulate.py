import cv2
import os
import time
import csv
import random
import numpy as np
import io
import math
from pymobility.models.mobility import random_waypoint
from lt.encode import encoder as lt_encoder
from lt.decode import LtDecoder, block_from_bytes
import tkinter as tk
from collections import defaultdict
from encode_video import encode_multiple_res_video


input_video_path = "input_video_10mb.mp4"
output_video_path = "output_adaptive.mp4"
block_size = 16
velocity = (1.0, 2.0)
output_csv = "video_simulation_results.csv"
frame_dump_dir = "frame_debug"
os.makedirs(frame_dump_dir, exist_ok=True)
per_frame_results_csv = "link_diagnostics_5mb.csv"

simulation_time_step = 0.1        
uav_speed = 8               
trials = 20         
bitrate_Mbps = 6

simulation_config = {
    "scenarios": {
        "UMa": {
            "name": "UMa",
            "altitudes_m": [35, 70],        
            "receiver_height_m": 25,         
            "trace_area": (500, 500),        
        },
        "InF": {
            "name": "InF",
            "altitudes_m": [8, 12],         
            "receiver_height_m": 2,          
            "trace_area": (500, 500),        
        },
    },

    "frequencies": {
        5.9: {
            "label": "5.9GHz",
            "freq_GHz": 5.9,
            "bandwidth_Hz": 20 * (10**6),            
        },
        30: {
            "label": "30GHz",
            "freq_GHz": 30.0,
            "bandwidth_Hz": 100 * (10**6),          
        },
    },

    "channel_conditions": ["LOS_ONLY", "LOS_NLOS"]
}


def build_sim_cfg(scenario_key, altitude_m, freq_key, channel_condition, base_flags=None):
    cfg = simulation_config  
    scen = cfg["scenarios"][scenario_key]
    band = cfg["frequencies"][freq_key]

    sim_cfg = {
        "scenario": scenario_key,                    
        "channel_condition": channel_condition,      
        "freq_GHz": band["freq_GHz"],
        "bandwidth_Hz": band["bandwidth_Hz"],
        "uav_altitude": altitude_m,
        "receiver_height": scen["receiver_height_m"],
        "trace_area": scen["trace_area"],
        "range_bin_size": 10
    }

    if scenario_key == "UMa" and float(band["freq_GHz"]) == 5.9:
        sim_cfg["trace_area"] = (2000, 2000)
        sim_cfg["range_bin_size"] = 100

    if base_flags:
        sim_cfg.update(base_flags)
    return sim_cfg


def update_uav_on_canvas(x, y):
    canvas.create_oval(x-1, y-1, x+1, y+1, fill="gray", outline="gray", tags="trail")

    canvas.delete("uav")
    canvas.create_oval(x-6, y-6, x+6, y+6, fill="red", tags="uav")

    canvas.update()


def compute_distance(pos1, pos2, sim_cfg):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (sim_cfg["uav_altitude"] - sim_cfg["receiver_height"])**2)

def compute_horizontal_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def path_loss_LOS(distance, freq_GHz, scenario):
    if(scenario == "UMa"):
        return 28 + 22 * math.log10(distance) + 20 * math.log10(freq_GHz)
    if(scenario == "InF"):
        return 31.84 + 21.5 * math.log10(distance) + 19 * math.log10(freq_GHz)

def path_loss_NLOS(distance, receiver_height, freq_GHz, scenario):
    if(scenario == "UMa"):
        return -17.5 + (46 - 7 * math.log10(receiver_height)) * math.log10(distance) + 20 * math.log10((40 * math.pi * freq_GHz) / 3)
    if(scenario == "InF"):
        return 18.6 + 35.7 * math.log10(distance) + 20 * math.log10(freq_GHz)

def los_probability(horizontal_distance, scenario):
    if(scenario == "UMa"):
        if (horizontal_distance < 18):
            return 1.0
        return (18/horizontal_distance) + (1 - (18/horizontal_distance)) * math.exp(-horizontal_distance/63)
    
    if(scenario == "InF"):
        k_subsce = 2 / math.log(1 / (1 - 0.5)) 
        return math.exp(-horizontal_distance/ k_subsce)

def loss_rate(distance, horizontal_distance, sim_cfg):
    scenario = sim_cfg["scenario"]             
    freq_GHz = sim_cfg["freq_GHz"]            
    bandwidth_Hz = sim_cfg["bandwidth_Hz"]   
    receiver_height = sim_cfg["receiver_height"]
    channel_condition = sim_cfg["channel_condition"] 

    if channel_condition == "LOS_ONLY":
        is_los = True
    else:
        los_prob = los_probability(horizontal_distance, scenario)
        is_los = (random.random() < los_prob)

    if is_los:
        path_loss = path_loss_LOS(distance, freq_GHz, scenario)
    else:
        path_loss = path_loss_NLOS(distance, receiver_height, freq_GHz, scenario)

    transmit_power = 23
    transmit_antenna_gain = 0
    receive_antenna_gain = 0

    N0 =  4.0 * (10 ** -21)
    noise_figure_db = 5 

    received_power = transmit_power - path_loss + transmit_antenna_gain + receive_antenna_gain
    received_power_watts = (10 ** (received_power/10)) * (10 ** -3)
    
    total_noise_power = N0 * bandwidth_Hz * (10 ** (noise_figure_db / 10.0))
    linear_SNR = received_power_watts/total_noise_power

    BER = 0.5 * math.exp(-0.6 * linear_SNR)

    L = int(block_size) * 8
    loss_rate = 1 - ((1 - BER) ** L)

    return loss_rate, linear_SNR, BER

def generate_interpolated_trace(model, start_position, max_steps, time_step, speed_mps, sim_cfg):
    trace = []
    current_pos = start_position
    while len(trace) < max_steps:
        target_pos = next(model)[0]
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = compute_distance(current_pos, target_pos, sim_cfg)
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


def simulate_frame_transmission(frame_data, trace, symbols_per_step, receiver_position, sim_cfg):
    symbol_generator = lt_encoder(io.BytesIO(frame_data), block_size)
    decoder = LtDecoder()
    symbols_sent = 0
    symbols_received = 0
    distances = []

    snr_samples = []
    ber_samples = []
    loss_samples = []

    K = len(frame_data) // block_size + 1

    start_time = time.time()

    for pos_index, position in enumerate(trace):
        print("current position:", position)
        update_uav_on_canvas(*position)
        if decoder.is_done():
            break
        distance = compute_distance(position, receiver_position, sim_cfg)
        horizontal_distance = compute_horizontal_distance(position, receiver_position)
        distances.append(distance)
        loss_prob, snr_lin, ber_val = loss_rate(distance, horizontal_distance, sim_cfg)
        snr_samples.append(snr_lin)
        ber_samples.append(ber_val)
        loss_samples.append(loss_prob)

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
    snr_avg = float(np.mean(snr_samples)) if snr_samples else 0.0
    ber_avg = float(np.mean(ber_samples)) if ber_samples else 0.0
    loss_avg = float(np.mean(loss_samples)) if loss_samples else 0.0
    snr_avg_dB = (10*math.log10(snr_avg)) if snr_avg > 0 else -999.0
    plr_emp = 1.0 - (symbols_received / max(1, symbols_sent))

    if decoder.is_done():
        return decoder.bytes_dump(), symbols_sent, latency, avg_distance, effective_rate, position, ber_avg, loss_avg, snr_avg_dB, plr_emp
    else:
        return None, symbols_sent, latency, avg_distance, effective_rate, trace[-1], ber_avg, loss_avg, snr_avg_dB, plr_emp

def run_video_simulation(range_data, sim_cfg):
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
    model = random_waypoint(nr_nodes=1, dimensions=sim_cfg["trace_area"], velocity=velocity)
    current_position = next(model)[0]
    quality_index = 0
    
    global window, canvas
    if "window" not in globals():
        w, h = sim_cfg["trace_area"]
        window = tk.Tk()
        window.title("UAV Simulation")
        canvas = tk.Canvas(window, width=w, height=h, bg="white")
        canvas.pack()

    rx_x = sim_cfg["trace_area"][0] / 2
    rx_y = sim_cfg["trace_area"][1] / 2
    receiver_position = (rx_x, rx_y)
    sim_cfg["receiver_position"] = receiver_position
    canvas.delete("base")
    canvas.create_rectangle(rx_x-10, rx_y-10, rx_x+10, rx_y+10, fill="blue", tags="base")

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

        K = len(frame_bytes) // block_size + 1
        max_symbols = math.ceil(1.10 * K)
        max_trace_steps = max_symbols // symbols_per_step + 1
        trace = generate_interpolated_trace(model, current_position, max_trace_steps, simulation_time_step, uav_speed, sim_cfg)

        decoded_data, symbols, latency, avg_distance, eff_rate, current_position, ber_avg, loss_avg, snr_avg_dB, plr_emp = simulate_frame_transmission(frame_bytes, trace, symbols_per_step, receiver_position, sim_cfg)


        latency = max(latency, 1e-6)
        throughput = (len(frame_bytes) * 8) / (latency * 1_000_000)

        bin_size = float(sim_cfg.get("range_bin_size", 10))
        range_mean = round((avg_distance // bin_size) * bin_size + (0.5 * bin_size))

        range_data[range_mean].setdefault("snr_db", []).append(snr_avg_dB)
        range_data[range_mean].setdefault("ber", []).append(ber_avg)
        range_data[range_mean].setdefault("plr", []).append(plr_emp)
        range_data[range_mean].setdefault("latency_all_sec", []).append(latency)

        overhead = symbols / (K if K > 0 else 1)
        with open(per_frame_results_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                frame_count,
                sim_cfg["scenario"],
                sim_cfg["channel_condition"],
                sim_cfg["freq_GHz"],
                sim_cfg["uav_altitude"],
                range_mean,
                round(snr_avg_dB, 2),
                f"{ber_avg:.3e}",
                f"{loss_avg:.3e}",
                K,
                symbols,
                round(overhead, 3),
                round(latency, 6),
                int(decoded_data is not None),
                selected_res,
                f"{plr_emp:.3e}",
            ])


        if decoded_data is not None:

            range_data[range_mean]["symbols_needed"].append(symbols)
            range_data[range_mean]["effective_rate"].append(eff_rate)
            range_data[range_mean]["resolution"].append(selected_res)
            range_data[range_mean]["loss_prob"].append(loss_avg) 

            if latency > 0.01 and throughput < 1000:
                range_data[range_mean]["latency_sec"].append(latency)
                range_data[range_mean]["throughput_Mbps"].append(throughput)

        else:
            range_data[range_mean]["failures"] = range_data[range_mean].get("failures", 0) + 1

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
    if out is not None:
        out.release()

def run_multiple_trials(sim_cfg):
    range_data = defaultdict(lambda: defaultdict(list))

    for trial in range(trials):
        print(f"\n----- TRIAL {trial + 1} -----")
        run_video_simulation(range_data, sim_cfg)

    file_exists = os.path.exists(output_csv)
    need_header = (not file_exists) or (os.path.getsize(output_csv) == 0)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:                
                writer.writerow(["scenario", "condition", "freq_GHz", "uav_altitude_m", "range_mean", 
                "symbols_needed", "latency_sec", "latency_all_sec", "throughput_Mbps", "effective_rate", 
                "resolution", "success_rate", "avg_SNR_dB", "avg_BER", "avg_PER", "avg_PLR"])

        for range_mean in sorted(range_data.keys()):
            snr_list = range_data[range_mean].get("snr_db", [])
            ber_list = range_data[range_mean].get("ber", [])
            plr_list = range_data[range_mean].get("plr", [])
            avg_snr_db = round(np.mean(snr_list), 2) if snr_list else ""
            avg_ber    = f"{np.mean(ber_list):.3e}" if ber_list else ""
            avg_plr    = f"{np.mean(plr_list):.3e}" if plr_list else ""
            success = len(range_data[range_mean]["latency_sec"])
            fails = range_data[range_mean].get("failures", 0)
            total = success + fails
            success_rate = round(100.0 * success / total, 1) if total > 0 else 0.0
            lat_all = range_data[range_mean].get("latency_all_sec", [])
            lat_all_mean = round(np.mean(lat_all), 6) if lat_all else ""
            
            row = [
                        sim_cfg["scenario"],
                        sim_cfg["channel_condition"],
                        sim_cfg["freq_GHz"],
                        sim_cfg["uav_altitude"],
                        range_mean,
                        round(np.mean(range_data[range_mean]["symbols_needed"]), 2) if success > 0 else "",
                        round(np.mean(range_data[range_mean]["latency_sec"]), 6) if success > 0 else "",
                        lat_all_mean,
                        round(np.mean(range_data[range_mean]["throughput_Mbps"]), 2) if success > 0 else "",
                        round(np.mean(range_data[range_mean]["effective_rate"]), 4) if success > 0 else "",
                        max(set(range_data[range_mean]["resolution"]), key=range_data[range_mean]["resolution"].count) if success > 0 else "",
                        success_rate,
                        avg_snr_db, avg_ber, avg_plr, 
                        round(np.mean(range_data[range_mean]["loss_prob"]), 6) if success > 0 else ""
            ]
            writer.writerow(row)
            print(f"Range {range_mean}m → Samples: {len(range_data[range_mean]['latency_sec'])}")

    print(f"\nAll {trials} trials complete. Averaged metrics saved to {output_csv}")


if __name__ == "__main__":
     

    for path in [output_csv, per_frame_results_csv]:
        if os.path.exists(path):
            os.remove(path)

    with open(per_frame_results_csv, "w", newline="") as f:
        w = csv.writer(f)
        
        w.writerow([
            "frame_id","scenario","condition","freq_GHz","uav_altitude_m",
            "range_mean","avg_SNR_dB","avg_BER","avg_loss_prob",
            "K","symbols_sent","overhead","latency_sec","decoded","resolution",
            "PER_est","PLR_emp"
        ])
    
    encode_multiple_res_video(input_video_path, "res_videos/")

    for scenario_key, scen in simulation_config["scenarios"].items():
        for channel_condition in simulation_config["channel_conditions"]:
            for freq_key in simulation_config["frequencies"].keys():
                for altitude_m in scen["altitudes_m"]:
                    print(f"\n=== RUN: {scenario_key} | {channel_condition} | {freq_key} GHz | {altitude_m} m ===")
                    sim_cfg = build_sim_cfg(
                        scenario_key=scenario_key,
                        altitude_m=altitude_m,
                        freq_key=freq_key,
                        channel_condition=channel_condition,
                    )
                    run_multiple_trials(sim_cfg)
    window.mainloop()
