import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageOps, ImageDraw
from effects import (
    exponential_zoom,
    camera_transform,
    dissolve,
    ParticleField,
    QuantumField,
    additive_overlay,
    create_vignette,
    apply_vignette,
    fade_to_black
)

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

def render_scene(
    scene, width, height, fps, mode,
    next_scene=None, scene_start=0
):

    duration = scene["duration"]
    frame_count = round(duration * fps)

    images = [
        load_image(name, width, height, mode)
        for name in scene["images"]
    ]

    effect = scene.get("effect", "slow_push")
    transition = scene.get("transition", "cut")

    zoom_defaults = {
        "slow_push": 1.15,
        "subtle_push": 1.08,
        "accelerating_zoom": 5.0,
        "continuous_macro_zoom": 2.5,
        "atomic_particle_field": 1.4,
        "quantum_wave_field": 1.2,
        "cosmic_reveal": 1.8,
        "planetary_descent": 2.0,
        "slow_descent": 1.2
    }

    zoom_start = scene.get("zoom_start", 1.0)

    zoom_end = scene.get(
        "zoom_end",
        zoom_defaults.get(effect, 1.25)
    )

    focal_point = scene.get(
        "focal_point", [0.5, 0.5]
    )

    particles = ParticleField(
        count=600,
        seed=42
    )

    quantum = QuantumField(width, height)

    vignette = create_vignette(
        width, height, strength=0.12
    )

    def make_frame(
        active_scene, active_images, t, global_time
    ):

        active_effect = active_scene.get(
            "effect", "slow_push"
        )

        start = active_scene.get("zoom_start", 1.0)

        end = active_scene.get(
            "zoom_end",
            zoom_defaults.get(active_effect, 1.25)
        )

        focus = active_scene.get(
            "focal_point", [0.5, 0.5]
        )

        # Camera motion
        if active_effect in (
            "accelerating_zoom",
            "continuous_macro_zoom",
            "planetary_descent"
        ):

            progress = t

        else:

            progress = smoothstep(t)

        # Multiple images within a scene
        if len(active_images) > 1:

            position = t * (len(active_images) - 1)

            first_index = int(position)

            second_index = min(
                first_index + 1,
                len(active_images) - 1
            )

            local_t = position - first_index

            outgoing = camera_transform(
                active_images[first_index],
                exponential_zoom(start, end, local_t),
                focus
            )

            incoming = camera_transform(
                active_images[second_index],
                start,
                focus
            )

            frame = dissolve(
                outgoing,
                incoming,
                np.clip((local_t - 0.6) / 0.4, 0, 1)
            )

        else:

            zoom = exponential_zoom(
                start, end, progress
            )

            frame = camera_transform(
                active_images[0],
                zoom,
                focus
            )

        # Atomic environment
        if active_effect == "atomic_particle_field":

            field = particles.render(
                width,
                height,
                global_time,
                speed=0.04
            )

            frame = cv2.addWeighted(
                frame, 0.65,
                field, 0.8,
                0
            )

        # Quantum environment
        elif active_effect == "quantum_wave_field":

            waves = quantum.render(global_time)

            field = particles.render(
                width,
                height,
                global_time,
                speed=0.04
            )

            frame = cv2.addWeighted(
                frame, 0.3,
                waves, 0.9,
                0
            )

            frame = additive_overlay(
                frame,
                field,
                strength=0.35
            )

        # Quantum-to-cosmic reveal
        elif active_effect == "cosmic_reveal":

            field = particles.render(
                width,
                height,
                global_time,
                speed=0.04
            )

            waves = quantum.render(global_time)

            frame = dissolve(
                waves,
                frame,
                smoothstep(t)
            )

            frame = additive_overlay(
                frame,
                field,
                strength=0.55
            )

        # Planetary journey
        elif active_effect == "planetary_descent":

            field = particles.render(
                width,
                height,
                global_time,
                speed=0.04
            )

            frame = additive_overlay(
                frame,
                field,
                strength=0.35 * (1 - t) ** 2
            )

        return apply_vignette(frame, vignette)

    # Prepare the next scene for a continuous transition
    next_images = None

    if next_scene and transition in (
        "zoom_blend",
        "dissolve",
        "particle_morph"
    ):

        next_images = [
            load_image(
                next_scene["images"][0],
                width,
                height,
                mode
            )
        ]

    # Render each individual frame
    for index in range(frame_count):

        t = index / max(frame_count - 1, 1)

        global_time = scene_start + index / fps

        frame = make_frame(
            scene,
            images,
            t,
            global_time
        )

        # Blend into the next scene without
        # increasing the total film duration
        if next_images is not None:

            transition_seconds = min(1.2, duration / 3)

            transition_frames = max(
                2,
                round(transition_seconds * fps)
            )

            transition_start = (
                frame_count - transition_frames
            )

            if index >= transition_start:

                blend_progress = (
                    (index - transition_start)
                    / (transition_frames - 1)
                )

                incoming = make_frame(
                    next_scene,
                    next_images,
                    0.0,
                    global_time
                )

                frame = dissolve(
                    frame,
                    incoming,
                    blend_progress
                )

        # Final fade into darkness
        if transition == "fade_to_black" and t > 0.8:

            frame = fade_to_black(
                frame,
                (t - 0.8) / 0.2
            )

        # Progress reporting
        if (
            index % fps == 0
            or index == frame_count - 1
        ):

            print(
                f"{scene['id']}: "
                f"{index + 1}/{frame_count} frames "
                f"({100 * (index + 1) / frame_count:.1f}%)",
                flush=True
            )

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

        scenes = config["scenes"]
        scene_start = 0.0

        for index, scene in enumerate(scenes):

            next_scene = (
                scenes[index + 1]
                if index + 1 < len(scenes)
                else None
            )

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
                mode,
                next_scene=next_scene,
                scene_start=scene_start
            ):

                writer.append_data(frame)
                total_frames += 1

            scene_start += scene["duration"]

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
