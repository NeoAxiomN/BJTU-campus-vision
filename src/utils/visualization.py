from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def draw_boxes(
	image_path: Path,
	boxes: Sequence[Sequence[float]],
	color: str = "lime",
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image


def save_precision_plot(
	path: Path,
	class_name: str,
	ks: list[int],
	values: list[float],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(ks, values, marker="o")
    plt.ylim(0.0, 1.05)
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title(class_name)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_image_grid(
	path: Path,
	images: list[Image.Image],
	titles: list[str],
	cols: int = 3,
):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, image in enumerate(images):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(image)
        ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
