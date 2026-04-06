#!/usr/bin/env python3
"""
Convert EMNIST Digits IDX gzip files into MNIST-style CSV files.

Expected input files:
  data/EMNIST/digits/raw/emnist-digits-train-images-idx3-ubyte.gz
  data/EMNIST/digits/raw/emnist-digits-train-labels-idx1-ubyte.gz
  data/EMNIST/digits/raw/emnist-digits-test-images-idx3-ubyte.gz
  data/EMNIST/digits/raw/emnist-digits-test-labels-idx1-ubyte.gz

Outputs:
  emnist_digits_train.csv
  emnist_digits_test.csv

Each row:
  784 normalized pixel values in [-1, 1]
  10 one-hot label values
"""

import argparse
import csv
import gzip
import os
import struct

import numpy as np
from PIL import Image


TRAIN_IMAGES = "emnist-digits-train-images-idx3-ubyte.gz"
TRAIN_LABELS = "emnist-digits-train-labels-idx1-ubyte.gz"
TEST_IMAGES = "emnist-digits-test-images-idx3-ubyte.gz"
TEST_LABELS = "emnist-digits-test-labels-idx1-ubyte.gz"


def orient_like_mnist(image_arr):
    image = Image.fromarray(image_arr)
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).rotate(90, expand=False)
    return np.array(image, dtype=np.uint8)


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"Image payload size mismatch in {path}: got {data.size}, expected {expected}")
    return data.reshape(count, rows, cols)


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"Label payload size mismatch in {path}: got {data.size}, expected {count}")
    return data


def write_csv(images, labels, output_path, apply_orientation_fix):
    if len(images) != len(labels):
        raise ValueError(f"Image/label count mismatch for {output_path}")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for image, label in zip(images, labels):
            if apply_orientation_fix:
                image = orient_like_mnist(image)

            pixels = image.astype(np.float32).reshape(-1) / 127.5 - 1.0
            onehot = np.zeros(10, dtype=np.float32)
            onehot[int(label)] = 1.0
            writer.writerow(np.concatenate([pixels, onehot]))


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Run `make emnist-raw` first to download the EMNIST Digits gzip files."
        )


def main():
    parser = argparse.ArgumentParser(description="Convert EMNIST Digits gzip IDX files into CSV files.")
    parser.add_argument("--root", default="data/EMNIST/digits/raw", help="Directory containing the four EMNIST Digits gzip files")
    parser.add_argument("--train-out", default="emnist_digits_train.csv", help="Output CSV for the training split")
    parser.add_argument("--test-out", default="emnist_digits_test.csv", help="Output CSV for the test split")
    parser.add_argument(
        "--no-orientation-fix",
        action="store_true",
        help="Disable the EMNIST rotation/flip correction and export raw IDX orientation",
    )
    args = parser.parse_args()

    train_images_path = os.path.join(args.root, TRAIN_IMAGES)
    train_labels_path = os.path.join(args.root, TRAIN_LABELS)
    test_images_path = os.path.join(args.root, TEST_IMAGES)
    test_labels_path = os.path.join(args.root, TEST_LABELS)

    require_file(train_images_path)
    require_file(train_labels_path)
    require_file(test_images_path)
    require_file(test_labels_path)

    train_images = load_idx_images(train_images_path)
    train_labels = load_idx_labels(train_labels_path)
    test_images = load_idx_images(test_images_path)
    test_labels = load_idx_labels(test_labels_path)

    apply_orientation_fix = not args.no_orientation_fix
    write_csv(train_images, train_labels, args.train_out, apply_orientation_fix)
    write_csv(test_images, test_labels, args.test_out, apply_orientation_fix)

    print(f"Wrote {args.train_out} ({os.path.getsize(args.train_out):,} bytes)")
    print(f"Wrote {args.test_out} ({os.path.getsize(args.test_out):,} bytes)")
    print(f"Orientation fix: {'enabled' if apply_orientation_fix else 'disabled'}")


if __name__ == "__main__":
    main()
