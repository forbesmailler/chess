"""Export trained NNUE PyTorch model to binary format for C++ inference.

Binary format:
    [4 bytes] magic number ("NNUE")
    [4 bytes] version (uint32)
    [4 bytes] input_size (uint32)
    [4 bytes] hidden1_size (uint32)
    [4 bytes] hidden2_size (uint32)
    [4 bytes] output_size (uint32)
    [float32 arrays] W1, b1, W2, b2, W3, b3

Weight matrices are stored row-major: W1 is (input_size x hidden1_size),
meaning input_size rows of hidden1_size floats each.
"""

import argparse
import struct

import numpy as np
import torch

from train_nnue import NNUE


INPUT_SIZE = 768
HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 32
OUTPUT_SIZE = 3
VERSION = 1


def export_model(pytorch_path, output_path):
    model = NNUE()
    model.load_state_dict(torch.load(pytorch_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(output_path, "wb") as f:
        # Header
        f.write(b"NNUE")
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", INPUT_SIZE))
        f.write(struct.pack("<I", HIDDEN1_SIZE))
        f.write(struct.pack("<I", HIDDEN2_SIZE))
        f.write(struct.pack("<I", OUTPUT_SIZE))

        # Weights and biases
        # PyTorch Linear stores weight as (out_features, in_features)
        # C++ expects (in_features, out_features) row-major
        # So we transpose before writing

        w1 = model.fc1.weight.data.numpy().T  # (768, 256)
        b1 = model.fc1.bias.data.numpy()
        w2 = model.fc2.weight.data.numpy().T  # (256, 32)
        b2 = model.fc2.bias.data.numpy()
        w3 = model.fc3.weight.data.numpy().T  # (32, 3)
        b3 = model.fc3.bias.data.numpy()

        for arr in [w1, b1, w2, b2, w3, b3]:
            f.write(arr.astype(np.float32).tobytes())

    total_params = (
        INPUT_SIZE * HIDDEN1_SIZE + HIDDEN1_SIZE
        + HIDDEN1_SIZE * HIDDEN2_SIZE + HIDDEN2_SIZE
        + HIDDEN2_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
    )
    file_size = 24 + total_params * 4  # header + float32 params
    print(f"Exported NNUE model to {output_path}")
    print(f"  Architecture: {INPUT_SIZE} -> {HIDDEN1_SIZE} -> {HIDDEN2_SIZE} -> {OUTPUT_SIZE}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export NNUE PyTorch model to binary format for C++"
    )
    parser.add_argument(
        "--model", required=True, help="Path to PyTorch model (.pt)"
    )
    parser.add_argument(
        "--output", default="nnue.bin", help="Output binary file path"
    )
    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
