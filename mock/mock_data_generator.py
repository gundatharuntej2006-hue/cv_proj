import json
from pathlib import Path
import numpy as np
from PIL import Image


def generate_synthetic_cxr(output_path: str = "mock/sample_cxr.png", size=(512, 512)):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    arr[:] = 40
    y, x = np.ogrid[:size[0], :size[1]]
    center_y, center_x = size[0] // 2, size[1] // 2
    mask = ((x - center_x) ** 2 / (size[1] * 0.4) ** 2 + (y - center_y) ** 2 / (size[0] * 0.45) ** 2) <= 1
    arr[mask] = 160
    left_lung = ((x - center_x * 0.7) ** 2 / (size[1] * 0.15) ** 2 + (y - center_y * 0.95) ** 2 / (size[0] * 0.3) ** 2) <= 1
    right_lung = ((x - center_x * 1.3) ** 2 / (size[1] * 0.15) ** 2 + (y - center_y * 0.95) ** 2 / (size[0] * 0.3) ** 2) <= 1
    arr[left_lung] = 70
    arr[right_lung] = 70
    img = Image.fromarray(arr)
    img.save(p)
    return str(p)


if __name__ == "__main__":
    generate_synthetic_cxr()
