#!/usr/bin/env python3
"""
Generate binary test-sample files for the web Explore view from mnist_test.csv.

Output:
  web-app/src/assets/test_samples_1000.bin   (first 1000 samples, ~785 KB)
  web-app/src/assets/test_samples_10000.bin  (all 10000 samples, ~7.8 MB)

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

CSV_PATH = "mnist_test.csv"
OUT_DIR = "web-app/src/assets"
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


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run create_mnist_csv.py first.")
        sys.exit(1)

    samples = []
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            vals = [float(x) for x in row]
            pixel_floats = vals[:DIM]
            onehot = vals[DIM:DIM + NUM_CLASSES]
            label = onehot.index(max(onehot))
            # Normalized [-1, 1] → uint8 [0, 255] (round-trips exactly since source was uint8)
            pixels = [max(0, min(255, int(round((v + 1.0) * 127.5)))) for v in pixel_floats]
            samples.append((pixels, label))

    print(f"Parsed {len(samples)} samples from {CSV_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)
    write_bin(samples[:1000], os.path.join(OUT_DIR, "test_samples_1000.bin"))
    write_bin(samples, os.path.join(OUT_DIR, "test_samples_10000.bin"))


if __name__ == "__main__":
    main()
