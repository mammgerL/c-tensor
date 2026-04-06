#!/usr/bin/env python3
"""
Generate synthetic digit samples that mimic the playground canvas pipeline.

Design goal:
  1. Draw digits on a 280x280 black canvas with white "mouse-like" strokes.
  2. Apply the same crop/scale/centroid-centering logic used by PlaygroundView.
  3. Export a standalone CSV dataset without touching the original MNIST files.

Default output:
  generated/playground_handwritten/playground_handwritten_train.csv

Each CSV row matches mnist_train.csv:
  784 normalized pixel columns in [-1, 1] + 10 one-hot label columns.
"""

import argparse
import csv
import math
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CANVAS_SIZE = 280
GRID_SIZE = 28
UPSCALE = 4
TARGET_SIZE = 18

Point = Tuple[float, float]
Stroke = List[Point]


def _variant(*strokes: Sequence[Point]) -> List[Stroke]:
    return [list(stroke) for stroke in strokes]


DIGIT_VARIANTS = {
    0: [
        _variant(
            [(0.50, 0.12), (0.68, 0.14), (0.80, 0.30), (0.82, 0.52), (0.77, 0.74), (0.60, 0.88), (0.40, 0.90), (0.23, 0.76), (0.18, 0.50), (0.22, 0.28), (0.35, 0.14), (0.50, 0.12)],
        ),
        _variant(
            [(0.52, 0.10), (0.73, 0.18), (0.83, 0.40), (0.80, 0.68), (0.66, 0.86), (0.44, 0.90), (0.25, 0.78), (0.18, 0.56), (0.23, 0.30), (0.37, 0.14), (0.52, 0.10)],
        ),
    ],
    1: [
        _variant(
            [(0.34, 0.28), (0.48, 0.16), (0.50, 0.86)],
            [(0.36, 0.86), (0.66, 0.86)],
        ),
        _variant(
            [(0.44, 0.14), (0.50, 0.12), (0.50, 0.88)],
        ),
    ],
    2: [
        _variant(
            [(0.22, 0.24), (0.34, 0.12), (0.58, 0.12), (0.76, 0.20), (0.80, 0.34), (0.70, 0.48), (0.54, 0.58), (0.38, 0.70), (0.24, 0.86), (0.80, 0.86)],
        ),
        _variant(
            [(0.20, 0.28), (0.30, 0.14), (0.52, 0.10), (0.74, 0.18), (0.80, 0.34), (0.72, 0.46), (0.54, 0.60), (0.32, 0.82), (0.78, 0.82)],
        ),
    ],
    3: [
        _variant(
            [(0.24, 0.18), (0.42, 0.12), (0.66, 0.14), (0.78, 0.26), (0.70, 0.42), (0.48, 0.50), (0.68, 0.56), (0.80, 0.70), (0.68, 0.86), (0.44, 0.90), (0.24, 0.82)],
        ),
        _variant(
            [(0.24, 0.16), (0.46, 0.12), (0.72, 0.18), (0.74, 0.34), (0.54, 0.46), (0.72, 0.56), (0.76, 0.74), (0.58, 0.88), (0.28, 0.84)],
        ),
    ],
    4: [
        _variant(
            [(0.66, 0.14), (0.66, 0.88)],
            [(0.20, 0.58), (0.78, 0.58)],
            [(0.22, 0.60), (0.62, 0.18)],
        ),
        _variant(
            [(0.62, 0.12), (0.62, 0.86)],
            [(0.22, 0.54), (0.78, 0.54)],
            [(0.24, 0.56), (0.58, 0.16)],
        ),
    ],
    5: [
        _variant(
            [(0.74, 0.14), (0.28, 0.14), (0.26, 0.42), (0.54, 0.42), (0.72, 0.48), (0.80, 0.62), (0.76, 0.80), (0.58, 0.90), (0.28, 0.84)],
        ),
        _variant(
            [(0.76, 0.12), (0.30, 0.12), (0.28, 0.38), (0.52, 0.38), (0.72, 0.46), (0.80, 0.62), (0.74, 0.82), (0.52, 0.90), (0.24, 0.82)],
        ),
    ],
    6: [
        _variant(
            [(0.72, 0.18), (0.54, 0.12), (0.34, 0.22), (0.22, 0.44), (0.20, 0.66), (0.30, 0.82), (0.50, 0.88), (0.70, 0.82), (0.76, 0.66), (0.66, 0.52), (0.48, 0.48), (0.30, 0.56)],
        ),
        _variant(
            [(0.70, 0.16), (0.50, 0.12), (0.30, 0.26), (0.22, 0.50), (0.26, 0.74), (0.42, 0.88), (0.64, 0.86), (0.78, 0.72), (0.74, 0.54), (0.58, 0.46), (0.38, 0.54)],
        ),
    ],
    7: [
        _variant(
            [(0.22, 0.16), (0.78, 0.16), (0.46, 0.88)],
        ),
        _variant(
            [(0.20, 0.14), (0.80, 0.14)],
            [(0.74, 0.16), (0.42, 0.88)],
        ),
    ],
    8: [
        _variant(
            [(0.50, 0.48), (0.34, 0.42), (0.24, 0.28), (0.30, 0.14), (0.50, 0.10), (0.70, 0.14), (0.76, 0.28), (0.66, 0.42), (0.50, 0.48), (0.30, 0.58), (0.22, 0.74), (0.30, 0.88), (0.50, 0.92), (0.72, 0.88), (0.80, 0.74), (0.70, 0.58), (0.50, 0.48)],
        ),
        _variant(
            [(0.50, 0.50), (0.32, 0.40), (0.26, 0.22), (0.38, 0.10), (0.58, 0.10), (0.74, 0.20), (0.70, 0.38), (0.50, 0.50), (0.30, 0.62), (0.26, 0.80), (0.42, 0.92), (0.62, 0.90), (0.76, 0.78), (0.70, 0.62), (0.50, 0.50)],
        ),
    ],
    9: [
        _variant(
            [(0.70, 0.42), (0.62, 0.20), (0.46, 0.12), (0.28, 0.18), (0.22, 0.34), (0.28, 0.48), (0.46, 0.54), (0.68, 0.48), (0.74, 0.62), (0.68, 0.86)],
        ),
        _variant(
            [(0.72, 0.44), (0.66, 0.22), (0.50, 0.10), (0.30, 0.16), (0.22, 0.34), (0.30, 0.50), (0.50, 0.56), (0.70, 0.48), (0.74, 0.68), (0.60, 0.90)],
        ),
    ],
}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def sample_polyline(points: Sequence[Point], samples_per_seg: int = 18) -> List[Point]:
    if len(points) < 2:
        return list(points)

    out: List[Point] = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        for step in range(samples_per_seg):
            t = step / float(samples_per_seg)
            out.append(lerp(p0, p1, t))
    out.append(points[-1])
    return out


def jitter_stroke(points: Sequence[Point], point_jitter: float) -> List[Point]:
    jittered = []
    for idx, (x, y) in enumerate(points):
        edge_scale = 0.35 if idx == 0 or idx == len(points) - 1 else 1.0
        jittered.append((
            clamp(x + random.gauss(0.0, point_jitter * edge_scale), 0.05, 0.95),
            clamp(y + random.gauss(0.0, point_jitter * edge_scale), 0.05, 0.95),
        ))
    return jittered


def add_mouse_wobble(points: Sequence[Point], wobble: float) -> List[Point]:
    if len(points) < 2:
        return list(points)

    out: List[Point] = []
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        nx, ny = (0.0, 0.0) if seg_len == 0 else (-dy / seg_len, dx / seg_len)

        steps = max(2, int(seg_len * 120))
        for step in range(steps):
            t = step / float(steps)
            bx, by = lerp((x0, y0), (x1, y1), t)
            wave = math.sin((t + random.random() * 0.15) * math.pi * 2.0)
            offset = random.gauss(0.0, wobble) + wave * wobble * 0.45
            out.append((
                clamp(bx + nx * offset, 0.03, 0.97),
                clamp(by + ny * offset, 0.03, 0.97),
            ))
    out.append(points[-1])
    return out


def maybe_split_stroke(stroke: Stroke) -> List[Stroke]:
    if len(stroke) < 4 or random.random() > 0.18:
        return [stroke]

    split_at = random.randint(1, len(stroke) - 2)
    left = stroke[:split_at + 1]
    right = stroke[split_at:]
    return [left, right]


def transform_points(points: Sequence[Point], angle_deg: float, scale: float, shift_x: float, shift_y: float) -> List[Point]:
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    transformed = []
    for x, y in points:
        cx = (x - 0.5) * scale
        cy = (y - 0.5) * scale
        rx = cx * ca - cy * sa
        ry = cx * sa + cy * ca
        transformed.append((
            clamp(0.5 + rx + shift_x, 0.02, 0.98),
            clamp(0.5 + ry + shift_y, 0.02, 0.98),
        ))
    return transformed


def stroke_to_canvas(points: Sequence[Point]) -> List[Tuple[int, int]]:
    margin = CANVAS_SIZE * 0.12
    span = CANVAS_SIZE - margin * 2.0
    canvas_points = []
    for x, y in points:
        canvas_points.append((
            int(round(margin + x * span)),
            int(round(margin + y * span)),
        ))
    return canvas_points


def draw_digit_canvas(digit: int) -> np.ndarray:
    variant = random.choice(DIGIT_VARIANTS[digit])
    angle_deg = random.gauss(0.0, 9.0)
    scale = random.uniform(0.88, 1.08)
    shift_x = random.gauss(0.0, 0.04)
    shift_y = random.gauss(0.0, 0.05)
    line_width = random.uniform(11.0, 18.0)
    point_jitter = random.uniform(0.012, 0.040)
    wobble = random.uniform(0.002, 0.010)

    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(image)

    for stroke_template in variant:
        stroke_points = jitter_stroke(stroke_template, point_jitter)
        stroke_points = transform_points(stroke_points, angle_deg, scale, shift_x, shift_y)
        stroke_points = sample_polyline(stroke_points, samples_per_seg=random.randint(10, 18))
        stroke_points = add_mouse_wobble(stroke_points, wobble)

        for partial in maybe_split_stroke(stroke_points):
            canvas_points = stroke_to_canvas(partial)
            if len(canvas_points) == 1:
                x, y = canvas_points[0]
                r = int(round(line_width * 0.5))
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
            else:
                draw.line(canvas_points, fill=255, width=int(round(line_width)))

    if random.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.35, 0.9)))

    return np.array(image, dtype=np.uint8)


def preprocess_canvas_to_grid(canvas_arr: np.ndarray) -> np.ndarray:
    ys, xs = np.where(canvas_arr > 10)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    scale = TARGET_SIZE / float(max(bw, bh))
    scaled_w = max(1, int(round(bw * scale)))
    scaled_h = max(1, int(round(bh * scale)))

    crop = canvas_arr[min_y:max_y + 1, min_x:max_x + 1].astype(np.float32)
    total = float(crop.sum())
    if total <= 0.0:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    yy, xx = np.indices(crop.shape, dtype=np.float32)
    cx = float((xx * crop).sum() / total) * scale
    cy = float((yy * crop).sum() / total) * scale

    offset_x = int(round(14 - cx))
    offset_y = int(round(14 - cy))
    offset_x = max(0, min(GRID_SIZE - scaled_w, offset_x))
    offset_y = max(0, min(GRID_SIZE - scaled_h, offset_y))

    crop_img = Image.fromarray(crop.astype(np.uint8))
    resized = crop_img.resize((scaled_w * UPSCALE, scaled_h * UPSCALE), Image.Resampling.LANCZOS)

    mid_size = GRID_SIZE * UPSCALE
    mid = Image.new("L", (mid_size, mid_size), 0)
    mid.paste(resized, (offset_x * UPSCALE, offset_y * UPSCALE))
    mid = mid.filter(ImageFilter.GaussianBlur(radius=1.0))

    final = mid.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
    return np.array(final, dtype=np.uint8)


def generate_sample(digit: int) -> Tuple[np.ndarray, np.ndarray]:
    raw_canvas = draw_digit_canvas(digit)
    grid = preprocess_canvas_to_grid(raw_canvas)
    return raw_canvas, grid


def save_preview(preview_dir: str, preview_samples: List[Tuple[int, np.ndarray, np.ndarray]]) -> None:
    os.makedirs(preview_dir, exist_ok=True)

    cell_raw = 56
    raw_grid = Image.new("L", (cell_raw * 10, cell_raw * 10), 0)
    proc_grid = Image.new("L", (28 * 10, 28 * 10), 0)

    per_digit = {digit: [] for digit in range(10)}
    for digit, raw_canvas, grid in preview_samples:
        if len(per_digit[digit]) < 10:
            per_digit[digit].append((raw_canvas, grid))

    for digit in range(10):
        for idx, (raw_canvas, grid) in enumerate(per_digit[digit]):
            raw_img = Image.fromarray(raw_canvas).resize((cell_raw, cell_raw), Image.Resampling.BILINEAR)
            raw_grid.paste(raw_img, (idx * cell_raw, digit * cell_raw))
            proc_grid.paste(Image.fromarray(grid), (idx * 28, digit * 28))

            raw_img.save(os.path.join(preview_dir, f"digit_{digit}_{idx}_raw.png"))
            Image.fromarray(grid).save(os.path.join(preview_dir, f"digit_{digit}_{idx}_processed.png"))

    raw_grid.save(os.path.join(preview_dir, "grid_raw_canvas.png"))
    proc_grid.save(os.path.join(preview_dir, "grid_processed_28x28.png"))


def generate_dataset(count: int, output_path: str, preview_dir: str = None) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    preview_samples: List[Tuple[int, np.ndarray, np.ndarray]] = []

    per_digit = count // 10
    extra = count - per_digit * 10

    for digit in range(10):
        sample_count = per_digit + (1 if digit < extra else 0)
        for _ in range(sample_count):
            raw_canvas, grid = generate_sample(digit)
            pixels = grid.astype(np.float32).reshape(-1) / 127.5 - 1.0
            label = np.zeros(10, dtype=np.float32)
            label[digit] = 1.0
            rows.append((pixels, label))

            if preview_dir and len(preview_samples) < 500:
                preview_samples.append((digit, raw_canvas, grid))

    random.shuffle(rows)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for pixels, label in rows:
            writer.writerow(np.concatenate([pixels, label]))

    if preview_dir:
        save_preview(preview_dir, preview_samples)
        print(f"Preview saved to {preview_dir}")

    print(f"Generated {len(rows)} samples -> {output_path}")
    print(f"File size: {os.path.getsize(output_path):,} bytes")


def main() -> None:
    default_dir = os.path.join("generated", "playground_handwritten")
    default_output = os.path.join(default_dir, "playground_handwritten_train.csv")
    default_preview = os.path.join(default_dir, "preview")

    parser = argparse.ArgumentParser(
        description="Generate standalone synthetic handwritten digits for the playground input domain."
    )
    parser.add_argument("--count", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default=default_output, help="Output CSV path")
    parser.add_argument("--preview", action="store_true", help="Save preview images")
    parser.add_argument("--preview-dir", type=str, default=default_preview, help="Preview directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    preview_dir = args.preview_dir if args.preview else None
    generate_dataset(args.count, args.output, preview_dir)


if __name__ == "__main__":
    main()
