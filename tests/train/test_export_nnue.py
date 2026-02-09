"""Tests for engine/train/export_nnue.py."""

import io
import struct
import sys
from pathlib import Path

import numpy as np

# export_nnue.py uses `from train_nnue import ...` (sibling import),
# so we must add its directory to sys.path.
_train_dir = str(Path(__file__).resolve().parent.parent.parent / "engine" / "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

from export_nnue import VERSION, export_model, export_state_dict, main  # noqa: E402

from engine.train.train_nnue import (  # noqa: E402
    HIDDEN1_SIZE,
    HIDDEN2_SIZE,
    INPUT_SIZE,
    NNUE,
    OUTPUT_SIZE,
)


def _export_to_buffer(model):
    """Export model to an in-memory buffer and return its bytes."""
    buf = io.BytesIO()
    export_state_dict(model.state_dict(), buf)
    return buf.getvalue()


def test_export_header():
    model = NNUE()
    data = _export_to_buffer(model)

    magic = data[:4]
    version, inp, h1, h2, out = struct.unpack("<5I", data[4:24])

    assert magic == b"NNUE"
    assert version == VERSION
    assert inp == INPUT_SIZE
    assert h1 == HIDDEN1_SIZE
    assert h2 == HIDDEN2_SIZE
    assert out == OUTPUT_SIZE


def test_export_roundtrip():
    model = NNUE()
    data = _export_to_buffer(model)

    # Read back weights from binary
    offset = 24  # skip header
    layers = [model.fc1, model.fc2, model.fc3]
    for layer in layers:
        out_sz, in_sz = layer.weight.shape
        # Binary stores (in_sz x out_sz) row-major (transposed from PyTorch)
        nbytes = in_sz * out_sz * 4
        w = np.frombuffer(data[offset : offset + nbytes], dtype=np.float32)
        w = w.reshape(in_sz, out_sz)
        offset += nbytes

        nbytes = out_sz * 4
        b = np.frombuffer(data[offset : offset + nbytes], dtype=np.float32)
        offset += nbytes

        expected_w = layer.weight.data.numpy().T  # (in, out)
        expected_b = layer.bias.data.numpy()

        np.testing.assert_allclose(w, expected_w, atol=1e-6)
        np.testing.assert_allclose(b, expected_b, atol=1e-6)


def test_export_file_size():
    """Verify exported buffer size matches header + total parameters * 4 bytes."""
    model = NNUE()
    data = _export_to_buffer(model)

    total_params = sum(p.numel() for p in model.parameters())
    expected_size = 24 + total_params * 4
    assert len(data) == expected_size


def test_export_no_extra_bytes():
    """Exported buffer should contain exactly the header and weight data, nothing more."""
    model = NNUE()
    data = _export_to_buffer(model)

    # After reading header (24 bytes), we should be able to read all weight data
    offset = 24
    for layer in [model.fc1, model.fc2, model.fc3]:
        out_sz, in_sz = layer.weight.shape
        offset += in_sz * out_sz * 4  # weight
        offset += out_sz * 4  # bias
    assert offset == len(data)


def test_export_model_writes_file(tmp_path):
    """export_model writes a valid binary file from a saved PyTorch model."""
    import torch

    model = NNUE()
    pt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pt_path)
    out_path = tmp_path / "nnue.bin"
    export_model(str(pt_path), str(out_path))

    data = out_path.read_bytes()
    assert data[:4] == b"NNUE"
    total_params = sum(p.numel() for p in model.parameters())
    assert len(data) == 24 + total_params * 4


def test_export_model_prints_info(tmp_path, capsys):
    """export_model prints architecture, parameter count, and file size."""
    import torch

    model = NNUE()
    pt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pt_path)
    out_path = tmp_path / "nnue.bin"
    export_model(str(pt_path), str(out_path))

    output = capsys.readouterr().out
    assert "Exported NNUE model" in output
    assert str(INPUT_SIZE) in output
    assert str(HIDDEN1_SIZE) in output
    assert str(HIDDEN2_SIZE) in output
    assert str(OUTPUT_SIZE) in output
    assert "Total parameters" in output
    assert "File size" in output


def test_export_main(tmp_path, monkeypatch):
    """main() parses args and calls export_model."""
    import torch

    model = NNUE()
    pt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pt_path)
    out_path = tmp_path / "out.bin"
    monkeypatch.setattr(
        "sys.argv",
        ["export_nnue.py", "--model", str(pt_path), "--output", str(out_path)],
    )
    main()
    assert out_path.exists()
    assert out_path.read_bytes()[:4] == b"NNUE"


def test_export_state_dict_returns_param_count():
    """export_state_dict returns the total parameter count."""
    model = NNUE()
    buf = io.BytesIO()
    count = export_state_dict(model.state_dict(), buf)
    expected = sum(p.numel() for p in model.parameters())
    assert count == expected
