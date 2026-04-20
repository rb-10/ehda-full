
""""
Video Pre processing & optical image generation script
Author: Ben Wolf
Date: 06-03-2026

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_gray_frames(cap):
    frames = []

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    return frames


def max_pool_to_size(img, output_size):
    """
    Downsample an image to a desired resolution using max pooling.

    Parameters:
        img (np.ndarray): Input image (grayscale or color).
        output_size (tuple): Desired output resolution (height, width).

    Returns:
        np.ndarray: Downsampled image.
    """
    orig_h, orig_w = img.shape[:2]
    new_h, new_w = output_size

    # Compute pooling factors (ceil to ensure coverage)
    h_factor = int(np.ceil(orig_h / new_h))
    w_factor = int(np.ceil(orig_w / new_w))

    # Compute needed padding to make dimensions divisible by factors
    pad_h = h_factor * new_h - orig_h
    pad_w = w_factor * new_w - orig_w

    # Pad the image with minimum value (so max pooling is correct)
    if len(img.shape) == 2:  # grayscale
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=img.min())
        channels = 1
    else:  # color
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=[int(v) for v in img.min(axis=(0, 1))])
        channels = img.shape[2]

    # Create max pooling kernel
    kernel = np.ones((h_factor, w_factor), dtype=np.uint8)

    # Apply max pooling
    if channels == 1:
        dilated = cv2.dilate(img_padded, kernel)
        pooled = dilated[::h_factor, ::w_factor]
    else:
        pooled_channels = []
        for ch in cv2.split(img_padded):
            dilated = cv2.dilate(ch, kernel)
            pooled_channels.append(dilated[::h_factor, ::w_factor])
        pooled = cv2.merge(pooled_channels)

    # Crop to exact desired output size (in case of rounding)
    return pooled[:new_h, :new_w]

def temporal_median_background(frames):
    """
    Detects motion by subtracting a robust estimate of the static background.

    Why this works:
    - The median across time approximates the static background.
    - Moving objects appear in only a few frames, so they disappear in the median.
    - Subtracting the median highlights anything that deviates from the static scene.
    - Taking the maximum difference across frames highlights the strongest motion.
    """

    stack = np.stack(frames).astype(np.float32)

    # Estimate static background
    background = np.median(stack, axis=0)

    # Compute absolute difference to background
    diff = np.abs(stack - background)

    # Take strongest deviation across time
    motion = np.max(diff, axis=0)

    motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
    return motion.astype(np.uint8)

def temporal_variance_map(frames):
    """
    Highlights pixels whose intensity fluctuates over time.

    Why this works:
    - Static background pixels have nearly constant intensity → low variance.
    - Moving or flickering objects produce intensity changes → high variance.
    - This method is very sensitive to tiny moving elements such as particles.
    """

    stack = np.stack(frames).astype(np.float32)

    # Compute variance per pixel across time
    var_map = np.var(stack, axis=0)

    var_map = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX)
    return var_map.astype(np.uint8)

def frame_difference_motion(frames):
    """
    Detects motion by computing differences between consecutive frames.

    Why this works:
    - Static background cancels out when subtracting consecutive frames.
    - Moving objects produce local intensity changes between frames.
    - Taking the maximum difference across all pairs captures the strongest motion.
    """

    diffs = []

    for i in range(len(frames) - 1):
        d = cv2.absdiff(frames[i], frames[i + 1])
        diffs.append(d)

    motion = np.max(np.stack(diffs), axis=0)

    motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
    return motion.astype(np.uint8)

def optical_flow_map(frames):
    """
    Computes optical flow and visualizes the magnitude of motion.

    Why this works:
    - Optical flow estimates pixel-wise motion vectors between frames.
    - The magnitude of these vectors highlights where motion occurs.
    - Areas with stronger movement accumulate higher motion magnitude.
    - Taking the maximum across frames reveals any motion that occurred.
    """

    mags = []

    for i in range(len(frames) - 1):

        flow = cv2.calcOpticalFlowFarneback(
            frames[i],
            frames[i + 1],
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mags.append(mag)

    motion = np.max(np.stack(mags), axis=0)

    motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
    return motion.astype(np.uint8)

def temporal_dog_motion(frames):
    """
    Detects motion while suppressing noise using temporal differences and smoothing.

    Why this works:
    - Frame differencing highlights motion but also amplifies noise.
    - A small Gaussian blur suppresses sensor noise and single-pixel artifacts.
    - Averaging differences across frames keeps consistent motion while
      suppressing random fluctuations.
    """

    diffs = []

    for i in range(len(frames) - 1):
        d = cv2.absdiff(frames[i], frames[i + 1])

        # reduce sensor noise
        d = cv2.GaussianBlur(d, (3, 3), 0)

        diffs.append(d)

    motion = np.mean(np.stack(diffs), axis=0)

    motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
    return motion.astype(np.uint8)

def temporal_darkest_pixel(frames):
    """
    Temporal stacking using the minimum intensity per pixel.

    Why this works:
    - Each pixel stores the darkest value it ever had.
    - Dark particles passing through the pixel reduce the value.
    - Over time this accumulates evidence of dark moving particles.
    - Static background remains largely unchanged.
    """

    stack = np.stack(frames).astype(np.uint8)

    # Keep the darkest value observed at each pixel
    darkest = np.min(stack, axis=0)

    darkest = cv2.normalize(darkest, None, 0, 255, cv2.NORM_MINMAX)
    return darkest

def tiny_particle_detector(frames):
    """
    Detects very small moving particles using spatial bandpass filtering
    combined with temporal variance.

    Why this works:

    1. Small Gaussian blur preserves particle-scale structures.
    2. Large Gaussian blur estimates background illumination.
    3. Subtracting them creates a bandpass filter that keeps only
       structures within a certain size range (particles).
    4. Temporal variance highlights pixels that fluctuate over time,
       which is typical for moving particles.
    """

    filtered_frames = []

    for f in frames:

        # remove pixel noise
        small = cv2.GaussianBlur(f, (3,3), 0)

        # estimate illumination / large background structures
        large = cv2.GaussianBlur(f, (51,51), 0)

        # bandpass filter keeps particle-sized structures
        bandpass = cv2.subtract(small, large)

        filtered_frames.append(bandpass)

    stack = np.stack(filtered_frames).astype(np.float32)

    # temporal variance reveals fluctuating pixels
    var_map = np.var(stack, axis=0)

    var_map = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX)
    return var_map.astype(np.uint8)

def original_optical_flow(frames):
    flow_magnitude_sum = None
    prev_gray = frames[0]
    for curr_gray in frames:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Only keep magnitude (how much movement), not direction
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if flow_magnitude_sum is None:
            flow_magnitude_sum = np.zeros_like(magnitude)
        flow_magnitude_sum += magnitude

        prev_gray = curr_gray

    # --- Average and normalize flow to 0-255 ---
    flow_avg = flow_magnitude_sum / len(frames)
    return cv2.normalize(flow_avg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def particle_trajectory_image(frames):
    """
    Generates a particle trajectory image by accumulating motion over time.

    Why this works:

    - Frame differences highlight pixels that change between frames.
    - Moving particles cause consistent small differences.
    - Accumulating these differences across time reveals particle paths.
    - Even very faint particles become visible when their motion
      integrates over many frames.
    """

    accumulator = None

    for i in range(len(frames) - 1):

        # detect motion between frames
        diff = cv2.absdiff(frames[i], frames[i+1])

        # suppress sensor noise
        diff = cv2.GaussianBlur(diff, (3,3), 0)

        if accumulator is None:
            accumulator = diff.astype(np.float32)
        else:
            accumulator += diff

    # normalize accumulated motion
    traj = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)

    traj = traj.astype(np.uint8)

    # # optional visualization boost
    # traj = cv2.equalizeHist(traj)

    return traj

def lucky_particle_stack(frames, keep_ratio=0.5):
    """
    Detect extremely faint particles using lucky imaging + temporal stacking.

    Why this works:

    - Bandpass filtering isolates particle-sized structures.
    - Some frames contain clearer particle signals than others.
    - We score frames by high-frequency content (sharpness).
    - Only the best frames are stacked to amplify weak signals.
    """

    filtered = []
    scores = []

    for f in frames:

        # bandpass filter to keep particle-sized features
        small = cv2.GaussianBlur(f, (3,3), 0)
        large = cv2.GaussianBlur(f, (31,31), 0)
        bandpass = cv2.subtract(small, large)

        filtered.append(bandpass)

        # measure frame sharpness (variance of Laplacian)
        score = cv2.Laplacian(bandpass, cv2.CV_32F).var()
        scores.append(score)

    filtered = np.stack(filtered).astype(np.float32)
    scores = np.array(scores)

    # keep only best frames
    keep_n = max(1, int(len(frames) * keep_ratio))
    best_idx = np.argsort(scores)[-keep_n:]

    best_frames = filtered[best_idx]

    # average best frames
    stacked = np.mean(best_frames, axis=0)

    stacked = cv2.normalize(stacked, None, 0, 255, cv2.NORM_MINMAX)

    stacked = stacked.astype(np.uint8)

    return stacked

def read_vid_display_image(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = read_gray_frames(cap)

    # show the initial frame
    plt.imshow(frames[0], cmap='gray');
    plt.title("First frame  -  " + video_path, fontsize=10);
    plt.show()

    ## in case you want to show all methods to highlight changes
    # plt.imshow(temporal_median_background(frames)); plt.title("Temporal median background"); plt.show()
    # plt.imshow(temporal_variance_map(frames)); plt.title("Temporal variance map"); plt.show()
    # plt.imshow(frame_difference_motion(frames)); plt.title("frame difference motion"); plt.show()
    # plt.imshow(original_optical_flow(frames)); plt.title("original optical flow"); plt.show()
    # plt.imshow(optical_flow_map(frames)); plt.title("optical flow map"); plt.show()
    # plt.imshow(temporal_dog_motion(frames)); plt.title("temporal dog motion"); plt.show()
    # plt.imshow(temporal_darkest_pixel(frames)); plt.title("temporal darkest pixel"); plt.show()
    # plt.imshow(tiny_particle_detector(frames)); plt.title("tiny_particle detector"); plt.show()
    # plt.imshow(particle_trajectory_image(frames)); plt.title("particle trajectory image"); plt.show()
    # plt.imshow(lucky_particle_stack(frames)); plt.title("Lucky Particle Stack"); plt.show()

    ## suggestion to merge three methods into an rgb image.
    merged_image = cv2.merge(
        (temporal_median_background(frames), tiny_particle_detector(frames), original_optical_flow(frames)))
    plt.imshow(merged_image)
    plt.title("enhanced changes - " + video_path, fontsize=10);
    plt.show()

    plt.imshow(max_pool_to_size(merged_image, (256, 256)))
    plt.title("maxpooling downsized image")
    plt.show()

if __name__ == "__main__":
     #read_vid_display_image(r"C:\Users\HV\Desktop\bruno_work\EHDA Image Classificaton\ehda-image-classification\datasets\open_setup\videos\clips\intermitent/clip_001_054.mp4")
    # read_vid_display_image("datasets/videos/intermitent/clip_001_056.mp4")
    # read_vid_display_image("datasets/videos/dripping/clip_003_102.mp4")
    # read_vid_display_image("datasets/videos/multi_jet/clip_003_073.mp4")
    import os
    from pathlib import Path

    # Set input and output base directories
    input_base = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\filmhood\SPLIT CLIPS")
    output_base = Path(r"C:\Users\HV\Desktop\bruno_work\save_electrospray\filmhood\PROCESSED CLIPS")

    # Loop through each class folder
    for class_folder in input_base.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            output_class_dir = output_base / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            # Loop through each .mp4 file in the class folder
            for video_file in class_folder.glob('*.mp4'):
                out_img_path = output_class_dir / (video_file.stem + '.png')
                if out_img_path.exists():
                    print(f"Skipping {video_file}, output already exists.")
                    continue
                print(f"Processing {video_file}")
                cap = cv2.VideoCapture(str(video_file))
                frames = read_gray_frames(cap)
                if not frames:
                    print(f"No frames found in {video_file}")
                    continue
                # Use merged RGB image of three methods, then crop/resize to 256x256
                merged_image = cv2.merge(
                    (temporal_median_background(frames), tiny_particle_detector(frames), original_optical_flow(frames)))
                processed_img = max_pool_to_size(merged_image, (256, 256))
                # Save processed image with same name as video, but .png
                cv2.imwrite(str(out_img_path), processed_img)
                print(f"Saved {out_img_path}")