# Standardized image and DICOM I/O utilities.

import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image


def load_image_normalized(path: Union[str, Path], target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    # Loads a PNG/JPEG/DICOM radiograph, scales to [0.0, 1.0], and resizes.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".dcm":
        try:
            import pydicom
            dcm = pydicom.dcmread(p)
            arr = dcm.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        except ImportError:
            raise ImportError("pydicom is required to read .dcm files")
    else:
        img = Image.open(p).convert("L")

    img = img.resize(target_size, Image.Resampling.BILINEAR)
    norm = np.array(img, dtype=np.float32) / 255.0
    return norm


def save_image_grayscale(array: np.ndarray, save_path: Union[str, Path]) -> None:
    # Saves a normalized [0, 1] 2D array as a grayscale PNG.
    scaled = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(scaled)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
