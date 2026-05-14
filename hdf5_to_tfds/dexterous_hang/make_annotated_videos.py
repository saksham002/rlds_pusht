import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import io
import os

from solve_subtask_boundaries import detect_boundaries as _detect_boundaries_from_file

DATA_DIR = "/data/group_data/rl/dexterous_robot_data/real_hang_full_success_r5_hdf5"
OUT_DIR = "/home/saksham3/projects/AIRe/rlds_dataset_builder/hdf5_to_tfds/video_annotations"

SUBTASK_LABELS = [
    "S0: Grasp the hanger",
    "S1: Lift hanger off the rod",
    "S2: Pass hanger to left arm",
    "S3: Hook one side of shirt",
    "S4: Hook other side of shirt",
    "S5: Place hanger on the rod",
]
SUBTASK_COLORS = [
    (255, 100, 100),
    (255, 200, 80),
    (100, 200, 100),
    (100, 255, 200),
    (80, 200, 255),
    (200, 150, 255),
]
EPISODES = [1, 3, 17, 21, 34, 42, 49]

# Load fonts once
try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 16)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 13)
except OSError:
    try:
        FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        FONT = ImageFont.load_default()
        FONT_SM = FONT


def detect_boundaries(ep):
    """Detect subtask boundaries by delegating to the solver module."""
    ep_path = f"{DATA_DIR}/episode_{ep}.hdf5"
    bounds, n = _detect_boundaries_from_file(ep_path)
    return bounds, n


def get_subtask_index(step, boundaries):
    for i, b in enumerate(boundaries):
        if step < b:
            return i
    return len(boundaries)


BAR_H = 36


def render_overlay(img, step, subtask_idx, total_steps):
    """Add black bar with subtask label and step counter to a PIL image."""
    w, h = img.size
    new_img = Image.new("RGB", (w, h + BAR_H), (0, 0, 0))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)

    label = SUBTASK_LABELS[subtask_idx]
    time_str = f"Step {step}/{total_steps - 1}  ({step / 30:.1f}s)"

    draw.text((8, h + 2), label, fill = SUBTASK_COLORS[subtask_idx], font = FONT)
    draw.text((w - 200, h + 6), time_str, fill = (200, 200, 200), font = FONT_SM)

    return new_img


def make_video(ep):
    boundaries, n = detect_boundaries(ep)
    print(f"Episode {ep}: {n} frames @ 30FPS, boundaries = {boundaries}")

    f = h5py.File(f"{DATA_DIR}/episode_{ep}.hdf5", "r")
    n_orig = f["obses/state/timestamp"].shape[0]
    idx = np.arange(0, n_orig, 2)

    # Read all images at 30 FPS (left/top camera)
    raw_imgs = f["obses/images/left/top"]

    # Get frame size from first image
    first_img = Image.open(io.BytesIO(raw_imgs[0].tobytes()))
    w, h = first_img.size
    frame_w, frame_h = w, h + BAR_H

    out_path = f"{OUT_DIR}/episode_{ep}_annotated.mp4"

    frames = []
    for i, orig_idx in enumerate(idx):
        jpeg_bytes = raw_imgs[orig_idx].tobytes()
        img = Image.open(io.BytesIO(jpeg_bytes))
        subtask_idx = get_subtask_index(i, boundaries)
        frame = render_overlay(img, i, subtask_idx, n)
        frames.append(np.array(frame))

        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{n} frames rendered")

    f.close()

    imageio.mimsave(out_path, frames, format = "mp4", fps = 30, codec = "libx264", quality = 8)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB, {n / 30:.1f}s)")


if __name__ == "__main__":
    for ep in EPISODES:
        make_video(ep)
