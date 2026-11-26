import cv2
import numpy as np


# --------------------------------------------------
# MATHEMATICAL MOTION FUNCTIONS
# --------------------------------------------------

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def exponential_zoom(start, end, t):

    t = float(np.clip(t, 0.0, 1.0))

    return start * (end / start) ** t


# --------------------------------------------------
# ADVANCED CAMERA MOVEMENT
# --------------------------------------------------

def camera_transform(
    frame,
    zoom=1.0,
    focal_point=(0.5, 0.5)
):

    height, width = frame.shape[:2]

    zoom = max(1.0, float(zoom))

    center_x = focal_point[0] * width
    center_y = focal_point[1] * height

    center_x = np.clip(
        center_x,
        width / (2 * zoom),
        width - width / (2 * zoom)
    )

    center_y = np.clip(
        center_y,
        height / (2 * zoom),
        height - height / (2 * zoom)
    )

    matrix = np.array([
        [zoom, 0, width / 2 - zoom * center_x],
        [0, zoom, height / 2 - zoom * center_y]
    ], dtype=np.float32)

    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


# --------------------------------------------------
# CINEMATIC IMAGE TRANSITIONS
# --------------------------------------------------

def dissolve(first, second, progress):

    alpha = float(smoothstep(progress))

    return cv2.addWeighted(
        first,
        1.0 - alpha,
        second,
        alpha,
        0
    )


def zoom_blend(first, second, progress):

    t = float(np.clip(progress, 0, 1))

    outgoing = camera_transform(
        first,
        zoom=1.0 + 0.4 * t
    )

    incoming = camera_transform(
        second,
        zoom=1.0 + 0.12 * t
    )

    return dissolve(outgoing, incoming, t)


# --------------------------------------------------
# PROCEDURAL PARTICLE FIELD
# --------------------------------------------------

class ParticleField:

    def __init__(
        self,
        count=350,
        seed=42
    ):

        rng = np.random.default_rng(seed)

        self.x = rng.uniform(-1.8, 1.8, count)
        self.y = rng.uniform(-1.2, 1.2, count)
        self.z = rng.uniform(0.1, 1.0, count)

        self.brightness = rng.uniform(
            0.25, 1.0, count
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
            width / 2
            + self.x * width * 0.35 / depth
        )

        screen_y = (
            height / 2
            + self.y * height * 0.35 / depth
        )

        for i in range(len(depth)):

            x = int(screen_x[i])
            y = int(screen_y[i])

            if not (
                0 <= x < width
                and 0 <= y < height
            ):
                continue

            radius = int(
                np.clip(1.2 / depth[i], 1, 4)
            )

            intensity = int(
                np.clip(
                    self.brightness[i]
                    * 180 / depth[i],
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
                (x, y),
                radius,
                colour,
                -1,
                lineType=cv2.LINE_AA
            )

        return frame


# --------------------------------------------------
# QUANTUM-INSPIRED WAVE FIELD
# --------------------------------------------------

class QuantumField:

    def __init__(self, width, height):

        self.width = width
        self.height = height

        # Lower internal resolution for faster CPU rendering
        internal_width = max(1, width // 3)
        internal_height = max(1, height // 3)

        self.x, self.y = np.meshgrid(
            np.linspace(
                -3, 3, internal_width,
                dtype=np.float32
            ),
            np.linspace(
                -2, 2, internal_height,
                dtype=np.float32
            )
        )

    def render(self, time):

        wave_1 = np.sin(
            self.x * 4.2
            + self.y * 2.1
            - time * 1.3
        )

        wave_2 = np.sin(
            self.x * 2.8
            - self.y * 3.7
            + time * 0.9
        )

        wave_3 = np.cos(
            np.sqrt(
                self.x**2 + self.y**2
            ) * 5.0
            - time * 1.7
        )

        interference = (
            wave_1 + wave_2 + wave_3
        ) / 3.0

        intensity = np.maximum(
            interference, 0
        ) ** 5

        frame = np.stack([
            intensity * 75,
            intensity * 95,
            intensity * 140
        ], axis=-1)

        frame = np.clip(
            frame, 0, 255
        ).astype(np.uint8)

        frame = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR
        )

        return frame


# --------------------------------------------------
# LIGHT AND ENERGY OVERLAYS
# --------------------------------------------------

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


# --------------------------------------------------
# CINEMATIC VIGNETTE
# --------------------------------------------------

def create_vignette(
    width,
    height,
    strength=0.25
):

    x, y = np.meshgrid(
        np.linspace(
            -1, 1, width,
            dtype=np.float32
        ),
        np.linspace(
            -1, 1, height,
            dtype=np.float32
        )
    )

    distance = np.sqrt(x**2 + y**2)

    mask = 1.0 - strength * distance**2

    return np.clip(
        mask, 0, 1
    )[:, :, None]


def apply_vignette(frame, mask):

    return (
        frame.astype(np.float32) * mask
    ).astype(np.uint8)


# --------------------------------------------------
# FADE INTO DARKNESS
# --------------------------------------------------

def fade_to_black(frame, progress):

    brightness = 1.0 - float(
        smoothstep(progress)
    )

    return (
        frame.astype(np.float32)
        * brightness
    ).astype(np.uint8)
