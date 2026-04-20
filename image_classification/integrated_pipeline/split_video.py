import cv2
import os
from pathlib import Path

def split_video(folder, file_name):
    # -------- SETTINGS --------
    input_video = str(file_name)
    output_folder = folder / 'SPLIT CLIPS'
    frames_per_clip = 40
    # --------------------------

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print("Error opening video")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("Total frames:", total_frames)

    clip_index = 0
    frame_index = 0

    while True:
        frames = []

        # Collect frames for one clip
        for i in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) < frames_per_clip:
            break  # stop if not enough frames left

        # Use only the filename stem for output naming
        video_stem = Path(file_name).stem
        output_path = os.path.join(output_folder, f"clip_{video_stem}_{clip_index:03d}.mp4")

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

        print(f"Saved clip {clip_index}")
        clip_index += 1

    cap.release()
    return output_folder
    print("Done.")

if __name__ == "__main__":
    solution = "ethanol_hv_nozzle"
    folder = Path(rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\{solution}")
    for file_name in Path(folder / 'raw').glob('*.mp4'):
        split_video(folder, file_name)