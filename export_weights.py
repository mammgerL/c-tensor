#!/usr/bin/env python3
"""
Export mnist_mlp.bin weights to a format usable by the browser.
Outputs: web-app/public/weights.bin (raw binary, same format as original)
         web-app/public/weights.json (metadata only, for debugging)

The browser will fetch weights.bin via fetch(), then parse the header + floats.
"""
import struct
import json
import sys
import os

MODEL_PATH = "mnist_mlp.bin"
OUT_BIN = "web-app/public/weights.bin"
OUT_JSON = "web-app/public/weights.json"

HEADER_FMT = '<IIiiiiii'  # magic(u32), version(u32), w1_rows, w1_cols, b1_len, w2_rows, w2_cols, b2_len
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32 bytes

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run `make train-run` first.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        data = f.read()

    # Parse header
    header = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic, version, w1_rows, w1_cols, b1_len, w2_rows, w2_cols, b2_len = header

    print(f"Model: magic=0x{magic:08X}, version={version}")
    print(f"  W1: ({w1_rows}, {w1_cols})")
    print(f"  B1: ({b1_len},)")
    print(f"  W2: ({w2_rows}, {w2_cols})")
    print(f"  B2: ({b2_len},)")

    # Verify sizes
    expected_size = HEADER_SIZE + (w1_rows * w1_cols + b1_len + w2_rows * w2_cols + b2_len) * 4
    assert len(data) == expected_size, f"Size mismatch: {len(data)} != {expected_size}"

    # Copy binary as-is (browser will fetch it directly)
    os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
    with open(OUT_BIN, 'wb') as f:
        f.write(data)
    print(f"Wrote {OUT_BIN} ({os.path.getsize(OUT_BIN)} bytes)")

    # Also create a JSON metadata file for reference
    offset = HEADER_SIZE
    w1 = struct.unpack(f'<{w1_rows * w1_cols}f', data[offset:offset + w1_rows * w1_cols * 4])
    offset += w1_rows * w1_cols * 4
    b1 = struct.unpack(f'<{b1_len}f', data[offset:offset + b1_len * 4])
    offset += b1_len * 4
    w2 = struct.unpack(f'<{w2_rows * w2_cols}f', data[offset:offset + w2_rows * w2_cols * 4])
    offset += w2_rows * w2_cols * 4
    b2 = struct.unpack(f'<{b2_len}f', data[offset:offset + b2_len * 4])

    meta = {
        "header": {
            "magic": hex(magic),
            "version": version,
            "w1_shape": [w1_rows, w1_cols],
            "b1_shape": [b1_len],
            "w2_shape": [w2_rows, w2_cols],
            "b2_shape": [b2_len],
        },
        "stats": {
            "w1_min": min(w1), "w1_max": max(w1),
            "w2_min": min(w2), "w2_max": max(w2),
            "b1_min": min(b1), "b1_max": max(b1),
            "b2_min": min(b2), "b2_max": max(b2),
            "total_params": w1_rows * w1_cols + b1_len + w2_rows * w2_cols + b2_len,
        }
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    print(f"Total parameters: {meta['stats']['total_params']:,}")

if __name__ == '__main__':
    main()
