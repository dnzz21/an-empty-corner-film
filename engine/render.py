
"""
An Empty Corner
Cinematic Rendering Engine - Version 2.0

CPU-based cinematic film rendering using static image assets,
procedural animation and mathematical visual effects.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio

from PIL import Image, ImageOps

from effects import (
    smoothstep,
    cinematic_camera,
    SpatialDistortion,
    ParticleField,
    QuantumField,
    radial_motion_blur,
    additive_overlay,
    colour_grade,
    create_vignette,
    apply_vignette,
    apply_transition,
    fade_to_black,
    black_screen
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = ROOT / "config" / "scenes.json"
IMAGE_DIR = ROOT / "assets" / "images"
OUTPUT_DIR = ROOT / "output"


# ============================================================
# 2. RENDERING PROFILES
# ============================================================

RENDER_PROFILES = {
    "preview": {
        "width": 1280,
        "height": 720,
        "fps": 24,
        "quality": 7,
        "preset": "fast"
    },
    "final": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "quality": 9,
        "preset": "medium"
    }
}


# ============================================================
# 3. SUPPORTED EFFECTS AND TRANSITIONS
# ============================================================

SUPPORTED_EFFECTS = {
    "slow_push",
    "subtle_push",
    "accelerating_zoom",
    "continuous_macro_zoom",
    "organic_distortion",
    "reality_warp",
    "deep_space_push",
    "cosmic_reveal",
    "planetary_descent",
    "atmospheric_descent",
    "targeted_descent",
    "twilight_drift",
    "black_screen"
}

SUPPORTED_TRANSITIONS = {
    "cut",
    "none",
    "zoom_blend",
    "macro_morph",
    "spatial_morph",
    "light_emergence",
    "cosmic_morph",
    "planetary_match",
    "atmospheric_blend",
    "targeted_match",
    "fade_to_black",
    "dissolve",
    "particle_morph"
}


# ============================================================
# 4. CONFIGURATION AND VALIDATION
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def validate_config(config):

    project = config["project"]
    scenes = config["scenes"]

    expected_duration = project["duration"]

    actual_duration = sum(
        scene["duration"]
        for scene in scenes
    )

    print("\nValidating project configuration...")
    print(f"Expected duration: {expected_duration}s")
    print(f"Configured duration: {actual_duration}s")
    print(f"Number of scenes: {len(scenes)}")

    if abs(actual_duration - expected_duration) > 0.001:

        raise ValueError(
            "Scene durations do not match project duration."
        )

    missing_images = []
    invalid_images = []
    asset_names = set()
    scene_ids = set()

    for scene in scenes:

        scene_id = scene["id"]

        if scene_id in scene_ids:
            raise ValueError(
                f"Duplicate scene ID: {scene_id}"
            )

        scene_ids.add(scene_id)

        if scene["duration"] <= 0:
            raise ValueError(
                f"Invalid duration: {scene_id}"
            )

        effect = scene.get("effect", "slow_push")

        transition = scene.get(
            "transition",
            "cut"
        )

        if effect not in SUPPORTED_EFFECTS:

            raise ValueError(
                f"Unsupported effect: {effect}"
            )

        if transition not in SUPPORTED_TRANSITIONS:

            raise ValueError(
                f"Unsupported transition: {transition}"
            )

        transition_duration = scene.get(
            "transition_duration",
            0
        )

        if not 0 <= transition_duration <= scene["duration"]:

            raise ValueError(
                f"Invalid transition duration: {scene_id}"
            )

        images = scene.get("images", [])

        if effect == "black_screen":
            continue

        if len(images) != 1:

            raise ValueError(
                f"{scene_id} must contain exactly one image."
            )

        focal_point = scene.get(
            "focal_point",
            [0.5, 0.5]
        )

        if (
            len(focal_point) != 2
            or not all(0 <= v <= 1 for v in focal_point)
        ):

            raise ValueError(
                f"Invalid focal point: {scene_id}"
            )

        for image_name in images:

            asset_names.add(image_name)

            image_path = IMAGE_DIR / image_name

            if not image_path.is_file():

                missing_images.append(image_name)

            else:

                try:

                    with Image.open(image_path) as image:
                        image.verify()

                except Exception as error:

                    invalid_images.append(
                        f"{image_name}: {error}"
                    )

    if missing_images:

        raise FileNotFoundError(
            "Missing image assets:\n"
            + "\n".join(missing_images)
        )

    if invalid_images:

        raise ValueError(
            "Invalid image assets:\n"
            + "\n".join(invalid_images)
        )

    print(f"Validated image assets: {len(asset_names)}")
    print("Configuration validation successful.\n")


# ============================================================
# 5. IMAGE LOADING
# ============================================================

def load_image(image_name, width, height):

    image_path = IMAGE_DIR / image_name

    with Image.open(image_path) as source:

        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )

        return np.array(
            image,
            dtype=np.uint8
        )


def load_scene_image(scene, width, height):

    if scene.get("effect") == "black_screen":

        return black_screen(width, height)

    image_name = scene["images"][0]

    return load_image(
        image_name,
        width,
        height
    )


# ============================================================
# 6. SCENE COLOUR PROFILES
# ============================================================

def get_colour_profile(scene_id):

    profiles = {

        "scene_01": {
            "contrast": 1.03,
            "saturation": 0.92,
            "exposure": -0.04,
            "temperature": -0.12
        },

        "scene_02": {
            "contrast": 1.02,
            "saturation": 0.90,
            "exposure": -0.03,
            "temperature": -0.10
        },

        "scene_03": {
            "contrast": 1.04,
            "saturation": 0.95,
            "exposure": -0.03,
            "temperature": 0.05
        },

        "scene_04": {
            "contrast": 1.04,
            "saturation": 0.96,
            "exposure": 0.0,
            "temperature": 0.05
        },

        "scene_05": {
            "contrast": 1.03,
            "saturation": 0.95,
            "exposure": 0.0,
            "temperature": 0.02
        },

        "scene_06": {
            "contrast": 1.04,
            "saturation": 0.98,
            "exposure": -0.03,
            "temperature": 0.0
        },

        "scene_07": {
            "contrast": 1.05,
            "saturation": 0.92,
            "exposure": -0.04,
            "temperature": -0.10
        },

        "scene_08": {
            "contrast": 1.05,
            "saturation": 0.95,
            "exposure": -0.03,
            "temperature": -0.08
        },

        "scene_09": {
            "contrast": 1.04,
            "saturation": 0.96,
            "exposure": -0.02,
            "temperature": -0.05
        },

        "scene_10": {
            "contrast": 1.03,
            "saturation": 0.98,
            "exposure": 0.0,
            "temperature": -0.04
        },

        "scene_11": {
            "contrast": 1.03,
            "saturation": 1.0,
            "exposure": 0.0,
            "temperature": -0.03
        },

        "scene_12": {
            "contrast": 1.04,
            "saturation": 1.02,
            "exposure": 0.0,
            "temperature": 0.0
        },

        "scene_13": {
            "contrast": 1.03,
            "saturation": 0.98,
            "exposure": 0.0,
            "temperature": 0.0
        },

        "scene_14": {
            "contrast": 1.03,
            "saturation": 0.98,
            "exposure": 0.0,
            "temperature": -0.02
        },

        "scene_15": {
            "contrast": 1.02,
            "saturation": 0.88,
            "exposure": -0.12,
            "temperature": -0.18
        },

        "scene_16": {
            "contrast": 1.02,
            "saturation": 0.94,
            "exposure": -0.03,
            "temperature": -0.08
        },

        "scene_17": {
            "contrast": 1.02,
            "saturation": 0.96,
            "exposure": 0.0,
            "temperature": -0.03
        },

        "scene_18": {
            "contrast": 1.01,
            "saturation": 0.95,
            "exposure": 0.0,
            "temperature": 0.0
        },

        "scene_19": {
            "contrast": 1.02,
            "saturation": 0.96,
            "exposure": 0.0,
            "temperature": -0.03
        },

        "scene_20": {
            "contrast": 1.02,
            "saturation": 0.93,
            "exposure": -0.02,
            "temperature": -0.02
        }
    }

    return profiles.get(
        scene_id,
        {
            "contrast": 1.0,
            "saturation": 1.0,
            "exposure": 0.0,
            "temperature": 0.0
        }
    )


def prepare_scene_image(scene, width, height):

    image = load_scene_image(
        scene,
        width,
        height
    )

    if scene.get("effect") == "black_screen":

        return image

    profile = get_colour_profile(
        scene["id"]
    )

    return colour_grade(
        image,
        **profile
    )


# ============================================================
# 7. CAMERA MOTION PROFILES
# ============================================================

def get_motion_profile(effect):

    if effect == "accelerating_zoom":

        return "accelerating"

    if effect in {
        "continuous_macro_zoom",
        "organic_distortion",
        "reality_warp",
        "deep_space_push",
        "cosmic_reveal",
        "planetary_descent",
        "atmospheric_descent",
        "targeted_descent"
    }:

        return "continuous"

    return "smooth"


# ============================================================
# 8. SHARED RENDERING RESOURCES
# ============================================================

class RenderContext:

    def __init__(self, width, height, fps, mode):

        self.width = width
        self.height = height
        self.fps = fps
        self.mode = mode

        self.distortion = SpatialDistortion(
            width,
            height
        )

        self.particles = ParticleField(
            count=250,
            seed=42
        )

        self.quantum = QuantumField(
            width,
            height
        )

        self.vignette = create_vignette(
            width,
            height,
            strength=0.055
        )


# ============================================================
# 9. SCENE FRAME GENERATION
# ============================================================

def generate_scene_frame(
    scene,
    image,
    progress,
    global_time,
    context
):

    effect = scene.get(
        "effect",
        "slow_push"
    )

    if effect == "black_screen":

        return image.copy()

    focal_point = tuple(
        scene.get(
            "focal_point",
            [0.5, 0.5]
        )
    )

    zoom_start = scene.get(
        "zoom_start",
        1.0
    )

    zoom_end = scene.get(
        "zoom_end",
        1.2
    )

    motion = get_motion_profile(effect)

    frame = cinematic_camera(
        image,
        progress=progress,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        focal_point=focal_point,
        motion=motion
    )

    # --------------------------------------------------------
    # Organic movement inside microscopic matter
    # --------------------------------------------------------

    if effect == "organic_distortion":

        intensity = scene.get(
            "distortion_intensity",
            0.15
        )

        frame = context.distortion.organic(
            frame,
            time=global_time,
            intensity=intensity
        )

    # --------------------------------------------------------
    # Unfamiliar spatial geometry
    # --------------------------------------------------------

    elif effect == "reality_warp":

        intensity = scene.get(
            "distortion_intensity",
            0.25
        )

        frame = context.distortion.reality_warp(
            frame,
            time=global_time,
            intensity=intensity
        )

    # --------------------------------------------------------
    # Microscopic forward movement
    # --------------------------------------------------------

    elif effect == "continuous_macro_zoom":

        strength = (
            0.012
            + 0.015 * progress
        )

        frame = radial_motion_blur(
            frame,
            strength=strength,
            focal_point=focal_point,
            samples=3
        )

    # --------------------------------------------------------
    # Accelerating camera
    # --------------------------------------------------------

    elif effect == "accelerating_zoom":

        frame = radial_motion_blur(
            frame,
            strength=0.025 * progress,
            focal_point=focal_point,
            samples=3
        )

    # --------------------------------------------------------
    # Deep-space movement
    # --------------------------------------------------------

    elif effect == "deep_space_push":

        if scene["id"] == "scene_13":

            particles = context.particles.render(
                context.width,
                context.height,
                global_time,
                speed=0.035
            )

            frame = additive_overlay(
                frame,
                particles,
                strength=0.08
            )

    # --------------------------------------------------------
    # Descent towards Earth
    # --------------------------------------------------------

    elif effect in {
        "planetary_descent",
        "atmospheric_descent",
        "targeted_descent"
    }:

        frame = radial_motion_blur(
            frame,
            strength=0.012 * progress,
            focal_point=focal_point,
            samples=3
        )

    # --------------------------------------------------------
    # Cinematic finishing
    # --------------------------------------------------------

    frame = apply_vignette(
        frame,
        context.vignette
    )

    return frame


# ============================================================
# 10. SCENE RENDERING
# ============================================================

def render_scene(
    writer,
    scene,
    image,
    next_scene,
    next_image,
    scene_start,
    context
):

    duration = scene["duration"]

    frame_count = round(
        duration * context.fps
    )

    transition_type = scene.get(
        "transition",
        "cut"
    )

    transition_duration = scene.get(
        "transition_duration",
        0
    )

    transition_frames = min(
        frame_count,
        round(
            transition_duration * context.fps
        )
    )

    transition_start = (
        frame_count - transition_frames
    )

    focal_point = tuple(
        scene.get(
            "focal_point",
            [0.5, 0.5]
        )
    )

    scene_timer = time.perf_counter()

    print(
        f"\nRendering {scene['id']}: "
        f"{scene['name']}"
    )

    print(
        f"Duration: {duration}s | "
        f"Frames: {frame_count} | "
        f"Effect: {scene['effect']}"
    )

    # Prepare the exact opening frame of the next scene.
    # This ensures the transition ends at the same visual
    # state from which the next scene begins.

    next_opening_frame = None

    if (
        next_scene is not None
        and next_image is not None
        and transition_frames > 0
        and transition_type not in {
            "cut",
            "none",
            "fade_to_black"
        }
    ):

        next_opening_frame = generate_scene_frame(
            next_scene,
            next_image,
            progress=0.0,
            global_time=scene_start + duration,
            context=context
        )

    # Render each frame individually.

    for frame_index in range(frame_count):

        progress = (
            frame_index
            / max(1, frame_count - 1)
        )

        local_time = (
            frame_index / context.fps
        )

        global_time = (
            scene_start + local_time
        )

        frame = generate_scene_frame(
            scene,
            image,
            progress,
            global_time,
            context
        )

        # ----------------------------------------------------
        # Apply the outgoing transition
        # ----------------------------------------------------

        if (
            transition_frames > 0
            and frame_index >= transition_start
        ):

            transition_progress = (
                frame_index - transition_start
            ) / max(
                1,
                transition_frames - 1
            )

            if transition_type == "fade_to_black":

                frame = fade_to_black(
                    frame,
                    transition_progress
                )

            elif next_opening_frame is not None:

                frame = apply_transition(
                    frame,
                    next_opening_frame,
                    transition_type,
                    transition_progress,
                    focal_point=focal_point,
                    distortion=context.distortion,
                    time=global_time
                )

        writer.append_data(
            np.ascontiguousarray(frame)
        )

        # Progress logging

        if (
            (frame_index + 1) % context.fps == 0
            or frame_index + 1 == frame_count
        ):

            print(
                f"  {frame_index + 1}/{frame_count}"
                " frames rendered",
                flush=True
            )

    elapsed = (
        time.perf_counter() - scene_timer
    )

    print(
        f"Scene completed in {elapsed:.2f}s",
        flush=True
    )

    return frame_count


# ============================================================
# 11. MAIN FILM RENDERER
# ============================================================

def render_film(mode):

    start_time = time.perf_counter()

    config = load_config()

    validate_config(config)

    project = config["project"]
    scenes = config["scenes"]

    profile = RENDER_PROFILES[mode]

    width = profile["width"]
    height = profile["height"]
    fps = profile["fps"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{mode}_{project['output']}"
    )

    expected_frames = sum(
        round(scene["duration"] * fps)
        for scene in scenes
    )

    print("=" * 60)
    print(project["title"])
    print("=" * 60)
    print(f"Render mode: {mode}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame rate: {fps} FPS")
    print(f"Duration: {project['duration']} seconds")
    print(f"Expected frames: {expected_frames}")
    print(f"Output: {output_path}")
    print("Audio: disabled for visual integration testing")
    print("=" * 60)

    context = RenderContext(
        width,
        height,
        fps,
        mode
    )

    # Load only the current and next image.
    # Avoid retaining all full-resolution assets in memory.

    current_image = prepare_scene_image(
        scenes[0],
        width,
        height
    )

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=profile["quality"],
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_params=[
            "-preset",
            profile["preset"],
            "-movflags",
            "+faststart"
        ]
    )

    total_frames = 0
    scene_start = 0.0

    try:

        for index, scene in enumerate(scenes):

            next_scene = (
                scenes[index + 1]
                if index + 1 < len(scenes)
                else None
            )

            next_image = None

            if next_scene is not None:

                next_image = prepare_scene_image(
                    next_scene,
                    width,
                    height
                )

            rendered_frames = render_scene(
                writer,
                scene,
                current_image,
                next_scene,
                next_image,
                scene_start,
                context
            )

            total_frames += rendered_frames

            scene_start += scene["duration"]

            current_image = next_image

    finally:

        writer.close()

    if total_frames != expected_frames:

        raise RuntimeError(
            f"Frame count mismatch: "
            f"expected {expected_frames}, "
            f"rendered {total_frames}"
        )

    elapsed = time.perf_counter() - start_time

    file_size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print("\n" + "=" * 60)
    print("RENDER COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video duration: {total_frames / fps:.2f}s")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Rendering time: {elapsed:.2f}s")
    print("=" * 60)


# ============================================================
# 12. COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Render An Empty Corner"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "preview",
            "final",
            "validate"
        ],
        default="preview"
    )

    args = parser.parse_args()

    if args.mode == "validate":

        config = load_config()

        validate_config(config)

    else:

        render_film(args.mode)


if __name__ == "__main__":

    main()
