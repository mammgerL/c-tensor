#!/usr/bin/env python3
"""
Generate binary sample files for the web Explore view from a CSV dataset.

Default output:
  web-app/src/assets/test_samples_1000.bin
  web-app/src/assets/test_samples_10000.bin

Binary format (little-endian):
  u32  magic        = 0x54534E4D  ('MNST')
  u32  count
  u32  dim          = 784
  u32  num_classes  = 10
  for each sample:
    u8[784]  pixels (0-255)
    u8       label  (0-9)
"""
import csv
import struct
import os
import sys
import argparse

CSV_PATH = "mnist_test.csv"
OUT_DIR = "web-app/src/assets"
PREFIX = "test_samples"
MAGIC = 0x54534E4D  # 'M','N','S','T' little-endian
DIM = 784
NUM_CLASSES = 10


def write_bin(samples, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, len(samples), DIM, NUM_CLASSES))
        for pixels, label in samples:
            f.write(bytes(pixels))
            f.write(bytes([label]))
    print(f"Wrote {path}  ({len(samples)} samples, {os.path.getsize(path):,} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a MNIST-format CSV dataset into Explore-view binary samples."
    )
    parser.add_argument("--input", default=CSV_PATH, help="Input CSV path")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    parser.add_argument("--prefix", default=PREFIX, help="Output filename prefix")
    parser.add_argument("--preview-count", type=int, default=1000, help="How many samples to store in the small preview bin")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    samples = []
    with open(args.input, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            vals = [float(x) for x in row]
            pixel_floats = vals[:DIM]
            onehot = vals[DIM:DIM + NUM_CLASSES]
            label = onehot.index(max(onehot))
            # Normalized [-1, 1] → uint8 [0, 255] (round-trips exactly since source was uint8)
            pixels = [max(0, min(255, int(round((v + 1.0) * 127.5)))) for v in pixel_floats]
            samples.append((pixels, label))

    print(f"Parsed {len(samples)} samples from {args.input}")
    os.makedirs(args.out_dir, exist_ok=True)
    small_count = min(args.preview_count, len(samples))
    write_bin(samples[:small_count], os.path.join(args.out_dir, f"{args.prefix}_{small_count}.bin"))
    write_bin(samples, os.path.join(args.out_dir, f"{args.prefix}_{len(samples)}.bin"))


if __name__ == "__main__":
    main()
