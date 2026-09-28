"""Fixed-severity image degradations. Images use OpenCV's BGR uint8 format."""

import cv2
import numpy as np


LEVELS = {
    "underexposure": {"mild": {"gamma": 1.5}, "medium": {"gamma": 2.0}, "severe": {"gamma": 3.0}},
    "overexposure": {"mild": {"gain": 1.25}, "medium": {"gain": 1.5}, "severe": {"gain": 2.0}},
    "low_contrast": {"mild": {"contrast": 0.75}, "medium": {"contrast": 0.50}, "severe": {"contrast": 0.25}},
    "gaussian_noise": {"mild": {"sigma": 8}, "medium": {"sigma": 16}, "severe": {"sigma": 25}},
    "defocus_blur": {"mild": {"radius": 3}, "medium": {"radius": 6}, "severe": {"radius": 10}},
    "motion_blur": {"mild": {"length": 7}, "medium": {"length": 15}, "severe": {"length": 25}},
    "resolution_loss": {"mild": {"scale": 0.75}, "medium": {"scale": 0.50}, "severe": {"scale": 0.25}},
    "jpeg_compression": {"mild": {"quality": 70}, "medium": {"quality": 40}, "severe": {"quality": 15}},
    "fog": {"mild": {"fog_coef": 0.2, "alpha_coef": 0.08}, "medium": {"fog_coef": 0.4, "alpha_coef": 0.10}, "severe": {"fog_coef": 0.6, "alpha_coef": 0.12}},
    "rain": {"mild": {"rain_type": "drizzle", "blur_value": 3, "brightness_coefficient": 0.90}, "medium": {"rain_type": "heavy", "blur_value": 5, "brightness_coefficient": 0.80}, "severe": {"rain_type": "torrential", "blur_value": 7, "brightness_coefficient": 0.70}},
}


def degrade(image: np.ndarray, kind: str, severity: str, seed: int) -> np.ndarray:
    params = LEVELS[kind][severity]
    pixels = image.astype(np.float32)
    if kind == "underexposure":
        result = 255.0 * np.power(pixels / 255.0, params["gamma"])
    elif kind == "overexposure":
        result = pixels * params["gain"]
    elif kind == "low_contrast":
        mean = float(pixels.mean())
        result = mean + params["contrast"] * (pixels - mean)
    elif kind == "gaussian_noise":
        result = pixels + np.random.default_rng(seed).normal(0, params["sigma"], image.shape)
    elif kind == "defocus_blur":
        radius = params["radius"]
        coordinates = np.arange(-radius, radius + 1)
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        kernel = (grid_x * grid_x + grid_y * grid_y <= radius * radius).astype(np.float32)
        kernel /= kernel.sum()
        result = cv2.filter2D(image, -1, kernel)
    elif kind == "motion_blur":
        length = params["length"]
        angle = np.random.default_rng(seed).uniform(0.0, np.pi)
        center = length // 2
        offset_x = center * np.cos(angle)
        offset_y = center * np.sin(angle)
        kernel = np.zeros((length, length), dtype=np.float32)
        start = (round(center - offset_x), round(center - offset_y))
        end = (round(center + offset_x), round(center + offset_y))
        cv2.line(kernel, start, end, 1.0, 1)
        kernel /= kernel.sum()
        result = cv2.filter2D(image, -1, kernel)
    elif kind == "resolution_loss":
        height, width = image.shape[:2]
        reduced = cv2.resize(image, (round(width * params["scale"]), round(height * params["scale"])), interpolation=cv2.INTER_AREA)
        result = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    elif kind == "jpeg_compression":
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, params["quality"]])
        if not success:
            raise RuntimeError("JPEG encoding failed")
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if result is None:
            raise RuntimeError("JPEG decoding failed")
    elif kind in {"fog", "rain"}:
        import albumentations as albumentations

        if kind == "fog":
            transform = albumentations.RandomFog(
                fog_coef_range=(params["fog_coef"], params["fog_coef"]),
                alpha_coef=params["alpha_coef"], p=1,
            )
        else:
            transform = albumentations.RandomRain(**params, p=1)
        pipeline = albumentations.Compose([transform], seed=seed)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(pipeline(image=rgb)["image"], cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unknown degradation: {kind}")
    return np.clip(result, 0, 255).astype(np.uint8)
