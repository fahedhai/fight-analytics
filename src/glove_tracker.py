"""
glove_tracker.py

Tracks a red boxing/Muay Thai glove across video frames using HSV color
segmentation, then detects strikes as peaks in the glove's movement speed.

Why color tracking instead of full-body pose estimation: this avoids
needing a large pre-trained pose model, and red gloves/wraps are common
enough in combat sports gyms that this generalizes to a lot of footage.
Trade-off: only tracks the glove, not the full skeleton, and can be
fooled by other red objects in frame (handled here with temporal
continuity -- we prefer the detection closest to the glove's last known
position over just picking the largest red blob every frame).

Run with:
    python glove_tracker.py --input path/to/video.mov --output_dir out/
"""

import cv2
import numpy as np
import argparse
import os
import json


def get_red_mask(hsv_frame):
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = mask1 | mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def track_glove(video_path, resize_width=480, min_area=150, max_jump_frac=0.25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = resize_width / orig_w
    resize_height = int(orig_h * scale)
    max_jump = max_jump_frac * resize_width

    positions = []  # (frame_idx, x, y) or (frame_idx, None, None) if not found
    last_pos = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (resize_width, resize_height))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = get_red_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [c for c in contours if cv2.contourArea(c) > min_area]

        chosen = None
        if candidates:
            if last_pos is not None:
                # Prefer the candidate closest to the last known position
                def dist_to_last(c):
                    M = cv2.moments(c)
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    return np.hypot(cx - last_pos[0], cy - last_pos[1])

                candidates_sorted = sorted(candidates, key=dist_to_last)
                nearest = candidates_sorted[0]
                if dist_to_last(nearest) < max_jump:
                    chosen = nearest
                else:
                    # nothing close enough -- re-acquire on the largest blob
                    chosen = max(candidates, key=cv2.contourArea)
            else:
                chosen = max(candidates, key=cv2.contourArea)

        if chosen is not None:
            M = cv2.moments(chosen)
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            positions.append((frame_idx, cx, cy))
            last_pos = (cx, cy)
        else:
            positions.append((frame_idx, None, None))

        frame_idx += 1

    cap.release()
    return positions, fps, resize_width, resize_height


def compute_speed(positions, fps):
    """Returns list of (frame_idx, speed_px_per_sec), interpolating gaps."""
    speeds = [0.0]
    for i in range(1, len(positions)):
        _, x0, y0 = positions[i - 1]
        _, x1, y1 = positions[i]
        if None in (x0, y0, x1, y1):
            speeds.append(0.0)
            continue
        dist = np.hypot(x1 - x0, y1 - y0)
        speeds.append(dist * fps)
    return speeds


def detect_strikes(speeds, min_speed_threshold, min_gap_frames=8):
    """Simple peak detection: a local max above threshold, with a minimum
    gap between detected strikes so one punch isn't counted multiple times."""
    strikes = []
    last_strike_frame = -min_gap_frames
    for i in range(1, len(speeds) - 1):
        if (
            speeds[i] > min_speed_threshold
            and speeds[i] >= speeds[i - 1]
            and speeds[i] >= speeds[i + 1]
            and (i - last_strike_frame) >= min_gap_frames
        ):
            strikes.append(i)
            last_strike_frame = i
    return strikes


def annotate_video(video_path, positions, strikes, output_path, resize_width, resize_height):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

    strike_set = set(strikes)
    strike_count = 0
    trail = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (resize_width, resize_height))

        _, x, y = positions[frame_idx]
        if x is not None:
            trail.append((int(x), int(y)))
            trail = trail[-15:]

        for pt in trail:
            cv2.circle(small, pt, 4, (0, 255, 255), -1)

        if x is not None:
            cv2.circle(small, (int(x), int(y)), 10, (0, 0, 255), 3)

        if frame_idx in strike_set:
            strike_count += 1
            cv2.putText(small, "STRIKE!", (resize_width // 2 - 70, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(small, f"Strikes: {strike_count}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(small)
        frame_idx += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--resize_width", type=int, default=480)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Tracking glove...")
    positions, fps, rw, rh = track_glove(args.input, resize_width=args.resize_width)

    print("Computing speed...")
    speeds = compute_speed(positions, fps)

    # Threshold: mean + 2 std of nonzero speeds, as a simple adaptive cutoff
    nonzero = [s for s in speeds if s > 0]
    threshold = np.mean(nonzero) + 1.5 * np.std(nonzero) if nonzero else 0

    print(f"Detecting strikes (speed threshold: {threshold:.0f}px/s)...")
    strikes = detect_strikes(speeds, threshold)
    print(f"Detected {len(strikes)} strikes")

    print("Rendering annotated video...")
    annotate_video(args.input, positions, strikes,
                    os.path.join(args.output_dir, "annotated.mp4"), rw, rh)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "num_strikes": len(strikes),
            "duration_sec": len(positions) / fps,
            "strikes_per_min": len(strikes) / (len(positions) / fps) * 60,
            "speed_threshold_px_per_sec": threshold,
        }, f, indent=2)

    print("Done. Output in", args.output_dir)
