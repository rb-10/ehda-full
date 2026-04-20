""""
Video Pre processing script
Author: Bruno Neves
Date: 06-03-2026

"""

#Before using Optical flow to transform a video into an Image, we are going to use 4 steps to try and improve our results. This Script is for that

#1st step, having a solid background, removing the gradient caused by the light
#2nd step, invert the img, black and white
#3rd step, CLAHE
#4th step, max pooling to make dropplets more vidible and reduce image size


import cv2
import numpy as np
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_BASE  = r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\datasets\open_setup\videos\clips"
OUTPUT_BASE = r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\datasets\open_setup\videos\resized clips"

TARGET_W = 256
TARGET_H = 256

# ── Tuning ─────────────────────────────────────────────────────────────────────

BLUR_KERNEL      = 51    # background estimation kernel
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
MIN_CONTOUR_AREA = 30    # minimum pixel area to keep as object (filters noise)


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide by blurred background to remove illumination gradient."""
    k    = BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    norm = cv2.divide(gray.astype(np.float32),
                      blur.astype(np.float32),
                      scale=128.0)
    return np.clip(norm, 0, 255).astype(np.uint8)

def invert(gray: np.ndarray) -> np.ndarray:
    """Invert: dark objects become bright, background becomes mid-gray."""
    return cv2.bitwise_not(gray)

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Boost local contrast so droplets stand out more."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)

def black_background(enhanced: np.ndarray) -> np.ndarray:
    """
    Use Otsu to find object locations, draw filled contours as a mask,
    then zero out everything outside — background becomes pure black
    without touching object pixel values.
    """
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    mask = np.zeros_like(enhanced)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    return cv2.bitwise_and(enhanced, mask)

def max_pool_resize(gray: np.ndarray) -> np.ndarray:
    """Resize using max pooling so bright object pixels win each block."""
    src_h, src_w = gray.shape
    kh = src_h // TARGET_H
    kw = src_w // TARGET_W
    crop = gray[: TARGET_H * kh, : TARGET_W * kw]
    r = crop.reshape(TARGET_H, kh, TARGET_W, kw)
    return r.max(axis=(1, 3)).astype(gray.dtype)


# ── Per-frame processing ───────────────────────────────────────────────────────

def process_frame(frame: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    gray     = flatten_background(gray)
    gray     = invert(gray)
    gray     = clahe.apply(gray)
    gray     = black_background(gray)
    gray     = max_pool_resize(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Single video ───────────────────────────────────────────────────────────────

def resize_video(video_path: str, output_path: str) -> None:
    """
    Process a single video and save the result.

    Args:
        video_path  : Full path to source .mp4.
        output_path : Full path for output file OR a directory.
                      If a directory, the output filename matches the input.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)

    if output_path.is_dir() or output_path.suffix.lower() != ".mp4":
        output_path = output_path / video_path.name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {video_path.name}  [{src_w}x{src_h} → {TARGET_W}x{TARGET_H}]  "
          f"{total_frames} frames @ {fps:.1f} fps")
    print(f"  Output → {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (TARGET_W, TARGET_H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process_frame(frame, clahe))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Done ({frame_idx} frames written)")


# ── Batch processing ───────────────────────────────────────────────────────────

def process_all_videos() -> None:
    """Process all .mp4 files under INPUT_BASE, mirroring folder structure."""
    input_base  = Path(INPUT_BASE)
    output_base = Path(OUTPUT_BASE)

    video_files = sorted(input_base.rglob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in: {input_base}")
        return

    print(f"Found {len(video_files)} video(s)\n")
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}]")
        output_path = output_base / video_path.relative_to(input_base)
        resize_video(str(video_path), str(output_path))
        print()

    print("All done!")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Single video test
    #video  = Path(r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\datasets\open_setup\videos\clips\cone_jet\clip_001_059.mp4")
    #output = Path(r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\datasets\open_setup\videos\resized clips")
    #resize_video(str(video), str(output))

    # Uncomment to run on all videos:
    process_all_videos()