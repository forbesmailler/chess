"""Tests for engine/train/export_nnue.py."""

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# export_nnue.py uses `from train_nnue import ...` (sibling import),
# so we must add its directory to sys.path.
_train_dir = str(Path(__file__).resolve().parent.parent.parent / "engine" / "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from export_nnue import VERSION, export_model  # noqa: E402

from engine.train.train_nnue import (  # noqa: E402
    HIDDEN1_SIZE,
    HIDDEN2_SIZE,
    INPUT_SIZE,
    NNUE,
    OUTPUT_SIZE,
)


def _save_random_model(path):
    """Save a randomly initialized NNUE model to disk."""
    model = NNUE()
    torch.save(model.state_dict(), path)
    return model


def test_export_header():
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = str(Path(tmp) / "model.pt")
        bin_path = str(Path(tmp) / "nnue.bin")
        _save_random_model(pt_path)
        export_model(pt_path, bin_path)

        with open(bin_path, "rb") as f:
            magic = f.read(4)
            version, inp, h1, h2, out = struct.unpack("<5I", f.read(20))

        assert magic == b"NNUE"
        assert version == VERSION
        assert inp == INPUT_SIZE
        assert h1 == HIDDEN1_SIZE
        assert h2 == HIDDEN2_SIZE
        assert out == OUTPUT_SIZE


def test_export_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = str(Path(tmp) / "model.pt")
        bin_path = str(Path(tmp) / "nnue.bin")
        model = _save_random_model(pt_path)
        export_model(pt_path, bin_path)

        # Read back weights from binary
        with open(bin_path, "rb") as f:
            f.read(24)  # skip header

            layers = [model.fc1, model.fc2, model.fc3]
            for layer in layers:
                out_sz, in_sz = layer.weight.shape
                # Binary stores (in_sz x out_sz) row-major (transposed from PyTorch)
                w = np.frombuffer(f.read(in_sz * out_sz * 4), dtype=np.float32)
                w = w.reshape(in_sz, out_sz)
                b = np.frombuffer(f.read(out_sz * 4), dtype=np.float32)

                expected_w = layer.weight.data.numpy().T  # (in, out)
                expected_b = layer.bias.data.numpy()

                np.testing.assert_allclose(w, expected_w, atol=1e-6)
                np.testing.assert_allclose(b, expected_b, atol=1e-6)


def test_export_file_size():
    """Verify exported file size matches header + total parameters * 4 bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = str(Path(tmp) / "model.pt")
        bin_path = str(Path(tmp) / "nnue.bin")
        model = _save_random_model(pt_path)
        export_model(pt_path, bin_path)

        total_params = sum(p.numel() for p in model.parameters())
        expected_size = 24 + total_params * 4
        actual_size = Path(bin_path).stat().st_size
        assert actual_size == expected_size


def test_export_no_extra_bytes():
    """Exported file should contain exactly the header and weight data, nothing more."""
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = str(Path(tmp) / "model.pt")
        bin_path = str(Path(tmp) / "nnue.bin")
        model = _save_random_model(pt_path)
        export_model(pt_path, bin_path)

        with open(bin_path, "rb") as f:
            data = f.read()

        # After reading header (24 bytes), we should be able to read all weight data
        offset = 24
        for layer in [model.fc1, model.fc2, model.fc3]:
            out_sz, in_sz = layer.weight.shape
            offset += in_sz * out_sz * 4  # weight
            offset += out_sz * 4  # bias
        assert offset == len(data)
