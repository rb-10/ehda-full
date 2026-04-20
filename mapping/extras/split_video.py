import cv2
import os
arr = [17, 18]
for i in range(10):
    # -------- SETTINGS --------
    file_name = "00"+str(i)
    input_video = rf"C:\Users\HV\Desktop\bruno_work\save_electrospray\Ethanol\raw\{file_name}.mp4"
    output_folder = r"C:\Users\HV\Desktop\bruno_work\save_electrospray\Ethanol\SPLIT CLIPS"
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

        # Collect 25 frames
        for i in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) < frames_per_clip:
            break  # stop if not enough frames left

        output_path = os.path.join(output_folder, f"clip_{file_name}_{clip_index:03d}.mp4")

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

        print(f"Saved clip {clip_index}")
        clip_index += 1

    cap.release()

    print("Done.")