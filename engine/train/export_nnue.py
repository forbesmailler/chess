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
from train_nnue import HIDDEN1_SIZE, HIDDEN2_SIZE, INPUT_SIZE, NNUE, OUTPUT_SIZE

VERSION = 1


def export_model(pytorch_path, output_path):
    model = NNUE()
    model.load_state_dict(
        torch.load(pytorch_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with open(output_path, "wb") as f:
        f.write(b"NNUE")
        f.write(
            struct.pack(
                "<5I", VERSION, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE
            )
        )

        # PyTorch Linear stores weight as (out, in); C++ expects (in, out) row-major
        for layer in [model.fc1, model.fc2, model.fc3]:
            f.write(layer.weight.data.numpy().T.astype(np.float32).tobytes())
            f.write(layer.bias.data.numpy().astype(np.float32).tobytes())

    total_params = sum(p.numel() for p in model.parameters())
    file_size = 24 + total_params * 4
    print(f"Exported NNUE model to {output_path}")
    print(
        f"  Architecture: {INPUT_SIZE} -> {HIDDEN1_SIZE} -> {HIDDEN2_SIZE} -> {OUTPUT_SIZE}"
    )
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export NNUE PyTorch model to binary format for C++"
    )
    parser.add_argument("--model", required=True, help="Path to PyTorch model (.pt)")
    parser.add_argument("--output", default="nnue.bin", help="Output binary file path")
    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
