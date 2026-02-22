"""Tests for engine/train/export_nnue.py."""

import io
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# export_nnue.py uses `from train_nnue import ...` (sibling import),
# so we must add its directory to sys.path.
_train_dir = str(Path(__file__).resolve().parent.parent.parent / "engine" / "train")
if _train_dir not in sys.path:
    sys.path.insert(0, _train_dir)

import export_nnue as _export_nnue_mod  # noqa: E402
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


def test_forward_pass_roundtrip():
    """Verify C++ forward pass (simulated) matches PyTorch output.

    Exports model to binary, reads weights back (with w2 transpose like C++),
    computes forward pass manually, and compares to PyTorch.
    """
    torch.manual_seed(42)
    model = NNUE()
    model.eval()

    data = _export_to_buffer(model)

    # Read weights from binary the way C++ does
    offset = 24
    # w1: stored as (INPUT_SIZE x HIDDEN1_SIZE) row-major
    w1_size = INPUT_SIZE * HIDDEN1_SIZE
    w1 = np.frombuffer(data[offset : offset + w1_size * 4], dtype=np.float32).reshape(
        INPUT_SIZE, HIDDEN1_SIZE
    )
    offset += w1_size * 4

    b1 = np.frombuffer(
        data[offset : offset + HIDDEN1_SIZE * 4], dtype=np.float32
    ).copy()
    offset += HIDDEN1_SIZE * 4

    # w2: stored as (HIDDEN1_SIZE x HIDDEN2_SIZE) row-major,
    # C++ transposes to (HIDDEN2_SIZE x HIDDEN1_SIZE) at load
    w2_size = HIDDEN1_SIZE * HIDDEN2_SIZE
    w2_raw = np.frombuffer(
        data[offset : offset + w2_size * 4], dtype=np.float32
    ).reshape(HIDDEN1_SIZE, HIDDEN2_SIZE)
    w2_t = w2_raw.T.copy()  # (HIDDEN2_SIZE x HIDDEN1_SIZE) like C++
    offset += w2_size * 4

    b2 = np.frombuffer(
        data[offset : offset + HIDDEN2_SIZE * 4], dtype=np.float32
    ).copy()
    offset += HIDDEN2_SIZE * 4

    w3 = np.frombuffer(
        data[offset : offset + HIDDEN2_SIZE * 4], dtype=np.float32
    ).copy()
    offset += HIDDEN2_SIZE * 4

    b3 = np.frombuffer(data[offset : offset + 4], dtype=np.float32).copy()
    offset += 4

    # Create sparse feature vector (simulating a real position)
    active_indices = [0, 65, 192, 324, 259, 384 + 0, 384 + 65, 768, 769, 770, 771]
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    for idx in active_indices:
        features[idx] = 1.0

    # PyTorch forward pass
    x_torch = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        pytorch_out = model(x_torch).item()

    # C++-style forward pass: sparse accumulation for layer 1
    h1 = b1.copy()
    for idx in active_indices:
        h1 += w1[idx]
    h1 = np.clip(h1, 0.0, 1.0)  # ClippedReLU

    # Layer 2: dense with transposed w2
    h2 = np.zeros(HIDDEN2_SIZE, dtype=np.float32)
    for j in range(HIDDEN2_SIZE):
        h2[j] = np.dot(w2_t[j], h1) + b2[j]
    h2 = np.clip(h2, 0.0, 1.0)  # ClippedReLU

    # Layer 3: single output with tanh
    logit = np.dot(w3, h2) + b3[0]
    cpp_out = float(np.tanh(logit))

    np.testing.assert_allclose(
        cpp_out,
        pytorch_out,
        atol=1e-5,
        err_msg=f"C++ forward pass ({cpp_out}) != PyTorch ({pytorch_out})",
    )


def test_export_model_cleans_up_output_on_error(tmp_path, monkeypatch):
    """If export_state_dict raises, export_model deletes the partial output file."""
    model = NNUE()
    pt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), pt_path)
    out_path = tmp_path / "nnue.bin"

    def _raise(state_dict, output):
        output.write(b"NNUE")  # partial write before failure
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(_export_nnue_mod, "export_state_dict", _raise)

    with pytest.raises(RuntimeError, match="simulated export failure"):
        _export_nnue_mod.export_model(str(pt_path), str(out_path))

    assert not out_path.exists()
