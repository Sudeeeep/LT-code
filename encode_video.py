import ffmpeg
import os

def encode_multiple_res_video(input_path, output_dir):
    resolutions = {
        '144p': (256, 144),
        '360p': (640, 360),
        '480p': (854, 480),
        '720p': (1280, 720)
    }

    os.makedirs(output_dir, exist_ok=True)

    for label, (w, h) in resolutions.items():
        output_path = os.path.join(output_dir, f'video_{label}.mp4')

        (
            ffmpeg
            .input(input_path)
            .output(output_path, vf=f'scale={w}:{h}', vcodec='libx264', acodec='aac', strict='experimental', preset='fast', crf=23)
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"Encoded {label} version at {output_path}")
