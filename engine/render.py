import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageOps, ImageDraw


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "scenes.json"
IMAGE_DIR = ROOT / "assets" / "images"
OUTPUT_DIR = ROOT / "output"


# --------------------------------------------------
# LOAD PROJECT CONFIGURATION
# --------------------------------------------------

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# TEMPORARY TEST IMAGE
# --------------------------------------------------

def create_placeholder(name, width, height):

    seed = sum(ord(char) for char in name)
    rng = np.random.default_rng(seed)

    x, y = np.meshgrid(
        np.linspace(-1, 1, width),
        np.linspace(-1, 1, height)
    )

    distance = np.sqrt(x**2 + y**2)

    glow = np.exp(-distance * 3)

    colour = rng.integers(60, 180, size=3)

    frame = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for channel in range(3):
        frame[:, :, channel] = (
            glow * colour[channel]
        ).astype(np.uint8)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    draw.text(
        (40, 40),
        f"TEST ASSET: {name}",
        fill=(255, 255, 255)
    )

    # Reference circles for checking camera movement
    for radius in [40, 100, 180, 260]:
        cx = width // 2
        cy = height // 2

        draw.ellipse(
            (
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius
            ),
            outline=(130, 150, 180),
            width=2
        )

    return np.array(image)


# --------------------------------------------------
# LOAD IMAGE ASSETS
# --------------------------------------------------

def load_image(filename, width, height, mode):

    path = IMAGE_DIR / filename

    if not path.exists():

        if mode == "final":
            raise FileNotFoundError(
                f"Required image is missing: {path}"
            )

        print(f"Using temporary asset: {filename}")

        return create_placeholder(
            filename, width, height
        )

    image = Image.open(path).convert("RGB")

    image = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS
    )

    return np.array(image)


# --------------------------------------------------
# CAMERA ZOOM
# --------------------------------------------------

def camera_zoom(frame, zoom, focal_point):

    height, width = frame.shape[:2]

    zoom = max(1.0, zoom)

    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))

    center_x = int(focal_point[0] * width)
    center_y = int(focal_point[1] * height)

    left = max(
        0,
        min(center_x - crop_width // 2,
            width - crop_width)
    )

    top = max(
        0,
        min(center_y - crop_height // 2,
            height - crop_height)
    )

    cropped = frame[
        top:top + crop_height,
        left:left + crop_width
    ]

    return cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )


# --------------------------------------------------
# SMOOTH MOTION
# --------------------------------------------------

def smoothstep(t):
    return t * t * (3 - 2 * t)


# --------------------------------------------------
# IMAGE BLENDING
# --------------------------------------------------

def blend_images(first, second, amount):

    amount = float(np.clip(amount, 0, 1))

    return cv2.addWeighted(
        first,
        1 - amount,
        second,
        amount,
        0
    )


# --------------------------------------------------
# RENDER INDIVIDUAL SCENE
# --------------------------------------------------

def render_scene(scene, width, height, fps, mode):

    duration = scene["duration"]
    frame_count = round(duration * fps)

    images = [
        load_image(name, width, height, mode)
        for name in scene["images"]
    ]

    zoom_start = scene.get("zoom_start", 1.0)
    zoom_end = scene.get("zoom_end", 1.25)

    focal_point = scene.get(
        "focal_point",
        [0.5, 0.5]
    )

    effect = scene.get("effect", "slow_push")

    for index in range(frame_count):

        t = index / max(frame_count - 1, 1)

        if effect == "accelerating_zoom":
            progress = t ** 2
        else:
            progress = smoothstep(t)

        zoom = zoom_start + (
            zoom_end - zoom_start
        ) * progress

        if len(images) == 1:

            frame = images[0]

        else:

            position = t * (len(images) - 1)

            first_index = int(position)

            second_index = min(
                first_index + 1,
                len(images) - 1
            )

            blend = smoothstep(
                position - first_index
            )

            frame = blend_images(
                images[first_index],
                images[second_index],
                blend
            )

        frame = camera_zoom(
            frame,
            zoom,
            focal_point
        )

        # Final fade into darkness
        if scene.get("transition") == "fade_to_black":

            if t > 0.8:

                fade = (1 - t) / 0.2

                frame = (
                    frame.astype(np.float32) * fade
                ).astype(np.uint8)

        yield frame


# --------------------------------------------------
# MAIN RENDERING PIPELINE
# --------------------------------------------------

def render_film(mode):

    config = load_config()
    project = config["project"]

    if mode == "preview":
        width = 1280
        height = 720
        fps = 24
    else:
        width = project["width"]
        height = project["height"]
        fps = project["fps"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / f"{mode}_{project['output']}"
    )

    print("=" * 50)
    print("AN EMPTY CORNER")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {project['duration']} seconds")

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_params=["-preset", "fast"]
    )

    total_frames = 0

    try:

        for scene in config["scenes"]:

            print(
                f"Rendering {scene['id']}: "
                f"{scene['name']}",
                flush=True
            )

            for frame in render_scene(
                scene,
                width,
                height,
                fps,
                mode
            ):

                writer.append_data(frame)
                total_frames += 1

    finally:
        writer.close()

    print("=" * 50)
    print("RENDER COMPLETE")
    print(f"Frames rendered: {total_frames}")
    print(f"Output: {output_path}")
    print("=" * 50)


# --------------------------------------------------
# COMMAND LINE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["preview", "final"],
        default="preview"
    )

    args = parser.parse_args()

    render_film(args.mode)
