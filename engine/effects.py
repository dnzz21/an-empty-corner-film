
"""
An Empty Corner
Cinematic Effects Engine - Version 2.0

CPU-based procedural cinematic rendering.

Dependencies:
    numpy
    opencv-python-headless
"""

import cv2
import numpy as np


# ============================================================
# 1. MOTION AND INTERPOLATION
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):
    return float(np.clip(value, minimum, maximum))


def smoothstep(t):
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t):
    t = clamp(t)
    return t ** 3 * (t * (t * 6 - 15) + 10)


def exponential_zoom(start, end, t):
    t = clamp(t)
    start = max(float(start), 0.001)
    end = max(float(end), 0.001)

    return start * (end / start) ** t


def interpolate(start, end, t):
    return start + (end - start) * clamp(t)


# ============================================================
# 2. CINEMATIC CAMERA
# ============================================================

def camera_transform(
    frame,
    zoom=1.0,
    focal_point=(0.5, 0.5),
    rotation=0.0,
    offset=(0.0, 0.0)
):
    height, width = frame.shape[:2]

    zoom = max(1.0, float(zoom))

    target_x = focal_point[0] * width
    target_y = focal_point[1] * height

    half_width = width / (2.0 * zoom)
    half_height = height / (2.0 * zoom)

    center_x = np.clip(
        target_x,
        half_width,
        width - half_width
    )

    center_y = np.clip(
        target_y,
        half_height,
        height - half_height
    )

    matrix = cv2.getRotationMatrix2D(
        (float(center_x), float(center_y)),
        rotation,
        zoom
    )

    matrix[0, 2] += (
        width / 2.0 - center_x + offset[0]
    )

    matrix[1, 2] += (
        height / 2.0 - center_y + offset[1]
    )

    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def cinematic_camera(
    frame,
    progress,
    zoom_start=1.0,
    zoom_end=1.2,
    focal_point=(0.5, 0.5),
    motion="smooth",
    rotation=0.0
):
    t = clamp(progress)

    if motion == "accelerating":
        movement = t ** 1.35

    elif motion == "continuous":
        movement = t

    elif motion == "decelerating":
        movement = 1.0 - (1.0 - t) ** 2

    else:
        movement = smootherstep(t)

    zoom = exponential_zoom(
        zoom_start,
        zoom_end,
        movement
    )

    return camera_transform(
        frame,
        zoom=zoom,
        focal_point=focal_point,
        rotation=rotation * movement
    )


# ============================================================
# 3. ORGANIC SPATIAL DISTORTION
# ============================================================

class SpatialDistortion:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.x, self.y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )

        self.nx = (
            self.x - width / 2.0
        ) / (width / 2.0)

        self.ny = (
            self.y - height / 2.0
        ) / (height / 2.0)

        self.radius = np.sqrt(
            self.nx ** 2 + self.ny ** 2
        )

        self.angle = np.arctan2(
            self.ny,
            self.nx
        )

    def organic(self, frame, time, intensity=0.2):

        intensity = clamp(intensity)

        wave_x = (
            np.sin(self.ny * 8.0 + time * 1.2)
            + 0.5 * np.sin(
                self.nx * 5.0 - time * 0.8
            )
        )

        wave_y = (
            np.cos(self.nx * 7.0 - time * 0.9)
            + 0.5 * np.sin(
                self.ny * 6.0 + time
            )
        )

        amplitude = min(
            self.width,
            self.height
        ) * 0.035 * intensity

        map_x = self.x + wave_x * amplitude
        map_y = self.y + wave_y * amplitude

        return cv2.remap(
            frame,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    def reality_warp(
        self,
        frame,
        time,
        intensity=0.3
    ):
        intensity = clamp(intensity)

        envelope = np.exp(
            -self.radius ** 2 * 1.4
        )

        twist = (
            np.sin(time * 0.65 + self.radius * 4.0)
            * intensity
            * envelope
            * 0.65
        )

        warped_angle = self.angle + twist

        warped_radius = self.radius * (
            1.0
            + 0.06
            * intensity
            * np.sin(
                self.radius * 10.0 - time * 1.5
            )
        )

        map_x = (
            np.cos(warped_angle)
            * warped_radius
            * self.width / 2.0
            + self.width / 2.0
        )

        map_y = (
            np.sin(warped_angle)
            * warped_radius
            * self.height / 2.0
            + self.height / 2.0
        )

        return cv2.remap(
            frame,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )


# ============================================================
# 4. CINEMATIC MOTION BLUR
# ============================================================

def radial_motion_blur(
    frame,
    strength=0.03,
    focal_point=(0.5, 0.5),
    samples=5
):
    strength = max(0.0, float(strength))
    samples = max(2, int(samples))

    if strength < 0.001:
        return frame

    result = frame.astype(np.float32)

    for index in range(1, samples):

        amount = index / (samples - 1)

        transformed = camera_transform(
            frame,
            zoom=1.0 + strength * amount,
            focal_point=focal_point
        )

        result += transformed.astype(np.float32)

    return np.clip(
        result / samples,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 5. BASIC IMAGE BLENDING
# ============================================================

def dissolve(first, second, progress):

    alpha = smoothstep(progress)

    return cv2.addWeighted(
        first,
        1.0 - alpha,
        second,
        alpha,
        0
    )


def additive_overlay(
    background,
    overlay,
    strength=0.5
):
    return cv2.addWeighted(
        background,
        1.0,
        overlay,
        float(strength),
        0
    )


# ============================================================
# 6. CINEMATIC TRANSITIONS
# ============================================================

def zoom_blend(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.35 * t,
        focal_point=focal_point
    )

    outgoing = radial_motion_blur(
        outgoing,
        strength=0.04 * np.sin(np.pi * t),
        focal_point=focal_point,
        samples=3
    )

    return dissolve(outgoing, second, t)


def macro_morph(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.5 * t,
        focal_point=focal_point
    )

    blend = dissolve(outgoing, second, t)

    return radial_motion_blur(
        blend,
        strength=0.07 * np.sin(np.pi * t),
        focal_point=focal_point,
        samples=4
    )


def spatial_morph(
    first,
    second,
    progress,
    distortion=None,
    time=0.0
):
    t = clamp(progress)

    blended = dissolve(first, second, t)

    if distortion is None:
        height, width = first.shape[:2]
        distortion = SpatialDistortion(width, height)

    intensity = 0.55 * np.sin(np.pi * t)

    return distortion.reality_warp(
        blended,
        time=time,
        intensity=intensity
    )


def light_emergence(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    height, width = first.shape[:2]

    x, y = np.meshgrid(
        np.linspace(0, 1, width, dtype=np.float32),
        np.linspace(0, 1, height, dtype=np.float32)
    )

    distance = np.sqrt(
        (x - focal_point[0]) ** 2
        + (y - focal_point[1]) ** 2
    )

    radius = 1.6 * smootherstep(t)

    mask = np.clip(
        (radius - distance) / 0.20,
        0.0,
        1.0
    )

    if t <= 0.0:
        return first.copy()

    if t >= 1.0:
        return second.copy()

    mask = mask[:, :, None]

    result = (
        first.astype(np.float32) * (1.0 - mask)
        + second.astype(np.float32) * mask
    )

    glow = np.exp(
        -distance ** 2 / 0.035
    )

    glow *= 16.0 * np.sin(np.pi * t)

    result += glow[:, :, None]

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def cosmic_morph(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.25 * t,
        focal_point=focal_point
    )

    result = dissolve(outgoing, second, t)

    return radial_motion_blur(
        result,
        strength=0.025 * np.sin(np.pi * t),
        focal_point=focal_point,
        samples=3
    )


def planetary_match(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.30 * t,
        focal_point=focal_point
    )

    return dissolve(outgoing, second, t)


def atmospheric_blend(
    first,
    second,
    progress
):
    t = clamp(progress)

    result = dissolve(first, second, t)

    haze_strength = (
        0.10 * np.sin(np.pi * t)
    )

    haze = np.full_like(
        result,
        (175, 195, 220)
    )

    result = cv2.addWeighted(
        result,
        1.0 - haze_strength,
        haze,
        haze_strength,
        0
    )

    return result


def targeted_match(
    first,
    second,
    progress,
    focal_point=(0.5, 0.5)
):
    t = clamp(progress)

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.45 * t,
        focal_point=focal_point
    )

    result = dissolve(outgoing, second, t)

    return radial_motion_blur(
        result,
        strength=0.04 * np.sin(np.pi * t),
        focal_point=focal_point,
        samples=4
    )


# ============================================================
# 7. PARTICLE FIELD
# ============================================================

class ParticleField:

    def __init__(self, count=350, seed=42):

        rng = np.random.default_rng(seed)

        self.x = rng.uniform(-1.8, 1.8, count)
        self.y = rng.uniform(-1.2, 1.2, count)
        self.z = rng.uniform(0.1, 1.0, count)

        self.brightness = rng.uniform(
            0.25,
            1.0,
            count
        )

    def render(
        self,
        width,
        height,
        time,
        speed=0.08
    ):
        frame = np.zeros(
            (height, width, 3),
            dtype=np.uint8
        )

        depth = (
            (self.z - time * speed) % 1.0
        ) + 0.08

        screen_x = (
            width / 2.0
            + self.x * width * 0.35 / depth
        )

        screen_y = (
            height / 2.0
            + self.y * height * 0.35 / depth
        )

        for x, y, z, brightness in zip(
            screen_x,
            screen_y,
            depth,
            self.brightness
        ):

            if 0 <= x < width and 0 <= y < height:

                radius = int(
                    np.clip(1.2 / z, 1, 3)
                )

                intensity = int(
                    np.clip(
                        brightness * 180 / z,
                        0,
                        255
                    )
                )

                colour = (
                    int(intensity * 0.75),
                    int(intensity * 0.85),
                    intensity
                )

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    radius,
                    colour,
                    -1,
                    cv2.LINE_AA
                )

        return frame


# ============================================================
# 8. QUANTUM WAVE FIELD
# ============================================================

class QuantumField:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.x, self.y = np.meshgrid(
            np.linspace(
                -3, 3, max(1, width // 3),
                dtype=np.float32
            ),
            np.linspace(
                -2, 2, max(1, height // 3),
                dtype=np.float32
            )
        )

    def render(self, time):

        wave1 = np.sin(
            self.x * 4.2
            + self.y * 2.1
            - time * 1.3
        )

        wave2 = np.sin(
            self.x * 2.8
            - self.y * 3.7
            + time * 0.9
        )

        wave3 = np.cos(
            np.sqrt(self.x ** 2 + self.y ** 2)
            * 5.0
            - time * 1.7
        )

        interference = (
            wave1 + wave2 + wave3
        ) / 3.0

        intensity = np.maximum(
            interference,
            0
        ) ** 5

        frame = np.stack(
            [
                intensity * 75,
                intensity * 95,
                intensity * 140
            ],
            axis=-1
        ).astype(np.uint8)

        return cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR
        )


# ============================================================
# 9. CINEMATIC COLOUR GRADING
# ============================================================

def colour_grade(
    frame,
    contrast=1.0,
    saturation=1.0,
    exposure=0.0,
    temperature=0.0
):
    image = frame.astype(np.float32) / 255.0

    image *= 2.0 ** exposure

    image = (
        (image - 0.5) * contrast
    ) + 0.5

    luminance = (
        image[:, :, 0] * 0.2126
        + image[:, :, 1] * 0.7152
        + image[:, :, 2] * 0.0722
    )

    luminance = luminance[:, :, None]

    image = (
        luminance
        + (image - luminance) * saturation
    )

    image[:, :, 0] += temperature * 0.035
    image[:, :, 2] -= temperature * 0.035

    return np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 10. VIGNETTE AND ATMOSPHERE
# ============================================================

def create_vignette(
    width,
    height,
    strength=0.25
):
    x, y = np.meshgrid(
        np.linspace(-1, 1, width, dtype=np.float32),
        np.linspace(-1, 1, height, dtype=np.float32)
    )

    distance = np.sqrt(x ** 2 + y ** 2)

    mask = np.clip(
        1.0 - strength * distance ** 2,
        0.0,
        1.0
    )

    return mask[:, :, None]


def apply_vignette(frame, mask):

    return np.clip(
        frame.astype(np.float32) * mask,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# 11. FADE AND BLACK SCREEN
# ============================================================

def fade_to_black(frame, progress):

    brightness = 1.0 - smootherstep(progress)

    return np.clip(
        frame.astype(np.float32) * brightness,
        0,
        255
    ).astype(np.uint8)


def fade_from_black(frame, progress):

    brightness = smootherstep(progress)

    return np.clip(
        frame.astype(np.float32) * brightness,
        0,
        255
    ).astype(np.uint8)


def black_screen(width, height):

    return np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )


# ============================================================
# 12. TRANSITION CONTROLLER
# ============================================================

def apply_transition(
    first,
    second,
    transition_type,
    progress,
    focal_point=(0.5, 0.5),
    distortion=None,
    time=0.0
):
    t = clamp(progress)

    if transition_type == "zoom_blend":
        return zoom_blend(
            first, second, t, focal_point
        )

    elif transition_type == "macro_morph":
        return macro_morph(
            first, second, t, focal_point
        )

    elif transition_type == "spatial_morph":
        return spatial_morph(
            first, second, t, distortion, time
        )

    elif transition_type == "light_emergence":
        return light_emergence(
            first, second, t, focal_point
        )

    elif transition_type == "cosmic_morph":
        return cosmic_morph(
            first, second, t, focal_point
        )

    elif transition_type == "planetary_match":
        return planetary_match(
            first, second, t, focal_point
        )

    elif transition_type == "atmospheric_blend":
        return atmospheric_blend(
            first, second, t
        )

    elif transition_type == "targeted_match":
        return targeted_match(
            first, second, t, focal_point
        )

    elif transition_type == "fade_to_black":
        return fade_to_black(first, t)

    elif transition_type == "dissolve":
        return dissolve(first, second, t)

    elif transition_type == "particle_morph":
        return spatial_morph(
            first, second, t, distortion, time
        )

    elif transition_type == "cut":
        return first if t < 1.0 else second

    elif transition_type == "none":
        return first

    else:
        raise ValueError(
            f"Unsupported transition: {transition_type}"
        )
