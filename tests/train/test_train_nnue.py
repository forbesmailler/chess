"""Tests for engine/train/train_nnue.py."""

import struct
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from engine.train.train_nnue import (
    NNUE,
    SelfPlayDataset,
    extract_features,
    main,
    train,
)

# --- extract_features ---


def _starting_placement():
    """Build 32-byte piece placement for the standard starting position.

    Board layout (rank 1 at bytes 0-3, rank 8 at bytes 28-31):
      sq 0=a1 is high nibble of byte 0, sq 1=b1 is low nibble of byte 0, etc.

    Piece encoding: 1-6 = white P/N/B/R/Q/K, 7-12 = black P/N/B/R/Q/K.
    """
    placement = bytearray(32)

    def set_sq(sq, nibble):
        byte_idx = sq // 2
        if sq % 2 == 0:
            placement[byte_idx] |= nibble << 4
        else:
            placement[byte_idx] |= nibble

    # Rank 1 (sq 0-7): R N B Q K B N R (white)
    rank1 = [4, 2, 3, 5, 6, 3, 2, 4]
    for i, p in enumerate(rank1):
        set_sq(i, p)

    # Rank 2 (sq 8-15): 8 white pawns
    for i in range(8, 16):
        set_sq(i, 1)

    # Ranks 3-6 empty

    # Rank 7 (sq 48-55): 8 black pawns
    for i in range(48, 56):
        set_sq(i, 7)

    # Rank 8 (sq 56-63): R N B Q K B N R (black)
    rank8 = [10, 8, 9, 11, 12, 9, 8, 10]
    for i, p in enumerate(rank8):
        set_sq(56 + i, p)

    return bytes(placement)


def test_extract_features_shape():
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0)
    assert features.shape == (773,)
    assert features.dtype == np.float32


def test_extract_features_start_position():
    placement = _starting_placement()
    features = extract_features(placement, side_to_move=0, castling=0b1111)
    # 32 pieces on the board = 32 piece features set
    piece_features = features[:768]
    assert piece_features.sum() == 32
    # All castling bits set
    assert features[768] == 1.0
    assert features[769] == 1.0
    assert features[770] == 1.0
    assert features[771] == 1.0


def test_extract_features_empty_board():
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0)
    assert features[:768].sum() == 0
    assert features[772] == 0.0


def test_extract_features_symmetry():
    """White vs black STM should swap own/opponent and mirror squares."""
    # Place a single white pawn (nibble 1) on a1 (sq 0)
    placement = bytearray(32)
    placement[0] = 0x10  # high nibble of byte 0 = sq 0 = nibble 1 (white pawn)

    feat_w = extract_features(bytes(placement), side_to_move=0)
    feat_b = extract_features(bytes(placement), side_to_move=1)

    # White STM: own pawn at sq 0 → index = 0*64 + 0 = 0
    assert feat_w[0] == 1.0
    # Black STM: opponent pawn, sq flipped (0^56=56) → index = 384 + 0*64 + 56 = 440
    assert feat_b[440] == 1.0


def test_extract_features_castling_white():
    placement = bytes(32)
    # Castling = 0b1010 → WK=1, WQ=0, BK=1, BQ=0
    features = extract_features(placement, side_to_move=0, castling=0b1010)
    assert features[768] == 1.0  # own kingside (WK)
    assert features[769] == 0.0  # own queenside (WQ)
    assert features[770] == 1.0  # opp kingside (BK)
    assert features[771] == 0.0  # opp queenside (BQ)


def test_extract_features_castling_black():
    placement = bytes(32)
    # Castling = 0b1010 → WK=1, WQ=0, BK=1, BQ=0
    # Black STM: own=black, opp=white
    features = extract_features(placement, side_to_move=1, castling=0b1010)
    assert features[768] == 1.0  # own kingside (BK)
    assert features[769] == 0.0  # own queenside (BQ)
    assert features[770] == 1.0  # opp kingside (WK)
    assert features[771] == 0.0  # opp queenside (WQ)


def test_extract_features_en_passant():
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0, en_passant_file=4)
    assert features[772] == 1.0

    features_no_ep = extract_features(placement, side_to_move=0, en_passant_file=255)
    assert features_no_ep[772] == 0.0


# --- NNUE model ---


def test_nnue_forward_shape():
    model = NNUE()
    x = torch.randn(4, 773)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4,)


def test_nnue_output_in_range():
    model = NNUE()
    x = torch.randn(8, 773)
    with torch.no_grad():
        out = model(x)
    assert (out >= -1).all()
    assert (out <= 1).all()


def test_nnue_output_scalar_per_sample():
    model = NNUE()
    x = torch.randn(8, 773)
    with torch.no_grad():
        out = model(x)
    assert out.dim() == 1
    assert out.shape[0] == 8


# --- SelfPlayDataset ---


def _make_synthetic_position(
    side_to_move=0,
    castling=0,
    en_passant_file=255,
    search_eval=0.5,
    game_result=1,
    ply=10,
):
    """Build a single 42-byte binary position."""
    buf = bytearray(42)
    # piece placement: all empty (32 zero bytes)
    buf[32] = side_to_move
    buf[33] = castling
    buf[34] = en_passant_file
    struct.pack_into("<f", buf, 35, search_eval)
    buf[39] = game_result
    struct.pack_into("<H", buf, 40, ply)
    return bytes(buf)


def test_dataset_from_binary():
    pos = _make_synthetic_position(game_result=2, search_eval=1.0)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name

    try:
        ds = SelfPlayDataset(tmp_path)
        assert len(ds) == 1

        features, target = ds[0]
        assert features.shape == (773,)
        assert target.shape == ()
        assert -1.0 <= target.item() <= 1.0
    finally:
        Path(tmp_path).unlink()


# --- train loop ---


def _make_training_data(path, num_positions=20):
    """Write synthetic binary training data to path."""
    with open(path, "wb") as f:
        for i in range(num_positions):
            result = i % 3  # cycle through loss/draw/win
            eval_val = (i - num_positions / 2) * 100.0
            f.write(
                _make_synthetic_position(
                    game_result=result,
                    search_eval=eval_val,
                    ply=i,
                )
            )


def _train_args(data_path, output_path=None, **overrides):
    """Build a Namespace mimicking argparse output for train()."""
    defaults = {
        "data": str(data_path),
        "output": str(output_path) if output_path else None,
        "epochs": 3,
        "batch_size": 8,
        "lr": 0.001,
        "patience": 10,
        "eval_weight": 0.75,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_train_returns_state_dict(tmp_path):
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)

    state = train(_train_args(data_file))

    assert "fc1.weight" in state
    assert "fc2.weight" in state
    assert "fc3.weight" in state


def test_train_model_loadable(tmp_path):
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)

    state = train(_train_args(data_file))

    model = NNUE()
    model.load_state_dict(state)
    x = torch.randn(1, 773)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,)
    assert -1.0 <= out.item() <= 1.0


def test_train_loss_decreases(tmp_path, capsys):
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file, num_positions=50)

    train(_train_args(data_file, epochs=10, patience=20))

    output = capsys.readouterr().out
    losses = []
    for line in output.splitlines():
        if "train_loss:" in line:
            loss_str = line.split("train_loss:")[1].split(",")[0].strip()
            losses.append(float(loss_str))

    assert len(losses) >= 2
    assert losses[-1] < losses[0]


def test_train_early_stopping(tmp_path, capsys):
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)

    train(_train_args(data_file, epochs=100, patience=2, lr=0.0))

    output = capsys.readouterr().out
    assert "Early stopping" in output
    # With lr=0 no improvement happens, should stop at epoch patience+1=3
    epoch_lines = [line for line in output.splitlines() if "Epoch" in line]
    assert len(epoch_lines) == 3


def test_train_respects_eval_weight(tmp_path):
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file, num_positions=30)

    state_a = train(_train_args(data_file, eval_weight=1.0, epochs=5))
    state_b = train(_train_args(data_file, eval_weight=0.0, epochs=5))

    # Different eval weights should produce different trained weights
    diff = (state_a["fc3.weight"] - state_b["fc3.weight"]).abs().sum()
    assert diff > 0


# --- SelfPlayDataset target blending ---


def test_dataset_target_win():
    """Game result = 2 (win) maps to result_scalar = +1."""
    pos = _make_synthetic_position(game_result=2, search_eval=0.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=0.0)  # pure result
        _, target = ds[0]
        assert abs(target.item() - 1.0) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_target_loss():
    """Game result = 0 (loss) maps to result_scalar = -1."""
    pos = _make_synthetic_position(game_result=0, search_eval=0.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=0.0)
        _, target = ds[0]
        assert abs(target.item() - (-1.0)) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_target_draw():
    """Game result = 1 (draw) maps to result_scalar = 0."""
    pos = _make_synthetic_position(game_result=1, search_eval=0.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=0.0)
        _, target = ds[0]
        assert abs(target.item()) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_target_pure_eval():
    """eval_weight=1.0 uses only search eval, ignoring result."""
    mate_value = 10000.0
    pos = _make_synthetic_position(game_result=2, search_eval=5000.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=1.0)
        _, target = ds[0]
        expected = np.clip(5000.0 / mate_value, -1.0, 1.0)
        assert abs(target.item() - expected) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_target_eval_clipping():
    """Search eval beyond MATE_VALUE gets clipped to [-1, 1]."""
    pos = _make_synthetic_position(game_result=1, search_eval=99999.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=1.0)
        _, target = ds[0]
        assert abs(target.item() - 1.0) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_target_blending():
    """Verify blending formula: eval_weight * eval + (1 - eval_weight) * result."""
    mate_value = 10000.0
    pos = _make_synthetic_position(game_result=2, search_eval=5000.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=0.75)
        _, target = ds[0]
        eval_scalar = np.clip(5000.0 / mate_value, -1.0, 1.0)
        result_scalar = 1.0  # win
        expected = 0.75 * eval_scalar + 0.25 * result_scalar
        assert abs(target.item() - expected) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_extract_features_black_piece_as_opponent():
    """Black knight (nibble 8) on b8 (sq 57) with white STM is opponent piece."""
    placement = bytearray(32)
    # sq 57 = byte 28, odd nibble
    placement[28] |= 8  # low nibble = 8 = black knight
    feat = extract_features(bytes(placement), side_to_move=0)
    # White STM, black knight is opponent. piece_type=1 (knight), sq=57
    # index = 384 + 1*64 + 57 = 505
    assert feat[505] == 1.0
    assert feat[:768].sum() == 1


def test_extract_features_black_piece_as_own():
    """Black knight (nibble 8) on b8 (sq 57) with black STM is own piece, sq flipped."""
    placement = bytearray(32)
    placement[28] |= 8  # sq 57 = black knight
    feat = extract_features(bytes(placement), side_to_move=1)
    # Black STM: own knight. piece_type=1, sq flipped: 57^56=1
    # index = 1*64 + 1 = 65
    assert feat[65] == 1.0
    assert feat[:768].sum() == 1


def test_extract_features_specific_piece_indices():
    """Verify exact feature indices for known pieces in starting position."""
    placement = _starting_placement()
    features = extract_features(placement, side_to_move=0, castling=0b1111)

    # White king on e1 (sq 4): own king = 5*64 + 4 = 324
    assert features[324] == 1.0
    # White queen on d1 (sq 3): own queen = 4*64 + 3 = 259
    assert features[259] == 1.0
    # Black king on e8 (sq 60): opponent king = 384 + 5*64 + 60 = 764
    assert features[764] == 1.0
    # Black queen on d8 (sq 59): opponent queen = 384 + 4*64 + 59 = 699
    assert features[699] == 1.0
    # White rook on a1 (sq 0): own rook = 3*64 + 0 = 192
    assert features[192] == 1.0


def test_dataset_target_eval_clipping_negative():
    """Search eval below -MATE_VALUE gets clipped to -1."""
    pos = _make_synthetic_position(game_result=1, search_eval=-99999.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=1.0)
        _, target = ds[0]
        assert abs(target.item() - (-1.0)) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_dataset_multiple_positions():
    """Dataset correctly indexes multiple positions."""
    positions = []
    for i in range(5):
        positions.append(
            _make_synthetic_position(
                game_result=i % 3, search_eval=float(i * 100), ply=i
            )
        )
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for pos in positions:
            f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path)
        assert len(ds) == 5
        for i in range(5):
            features, target = ds[i]
            assert features.shape == (773,)
            assert -1.0 <= target.item() <= 1.0
    finally:
        Path(tmp_path).unlink()


def test_extract_features_en_passant_file_zero():
    """En passant on a-file (file 0) should set feature 772."""
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0, en_passant_file=0)
    assert features[772] == 1.0


def test_extract_features_en_passant_file_seven():
    """En passant on h-file (file 7) should set feature 772."""
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0, en_passant_file=7)
    assert features[772] == 1.0


def test_nnue_single_sample():
    """NNUE forward pass with batch_size=1."""
    model = NNUE()
    x = torch.randn(1, 773)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1,)
    assert -1.0 <= out.item() <= 1.0


def test_dataset_target_half_blend():
    """eval_weight=0.5 blends eval and result equally."""
    mate_value = 10000.0
    # Win (result=2) with eval=2000 -> eval_scalar=0.2, result_scalar=1.0
    # target = 0.5*0.2 + 0.5*1.0 = 0.6
    pos = _make_synthetic_position(game_result=2, search_eval=2000.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=0.5)
        _, target = ds[0]
        expected = 0.5 * (2000.0 / mate_value) + 0.5 * 1.0
        assert abs(target.item() - expected) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_extract_features_no_en_passant_file_254():
    """File 254 is not a valid file but isn't 255; should still set feature 772."""
    placement = bytes(32)
    features = extract_features(placement, side_to_move=0, en_passant_file=254)
    assert features[772] == 1.0


def test_dataset_target_zero_eval():
    """Zero search eval with eval_weight=1.0 produces target 0."""
    pos = _make_synthetic_position(game_result=1, search_eval=0.0)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(pos)
        tmp_path = f.name
    try:
        ds = SelfPlayDataset(tmp_path, eval_weight=1.0)
        _, target = ds[0]
        assert abs(target.item()) < 1e-6
    finally:
        Path(tmp_path).unlink()


def test_train_saves_to_output_path(tmp_path):
    """When args.output is set, train() saves best model to that path."""
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)
    output_file = tmp_path / "model.pt"

    train(_train_args(data_file, output_path=output_file))
    assert output_file.exists()

    loaded = torch.load(output_file, map_location="cpu", weights_only=True)
    assert "fc1.weight" in loaded
    assert "fc3.weight" in loaded


def test_train_prints_saved_message(tmp_path, capsys):
    """When output is set, train() prints save messages."""
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)
    output_file = tmp_path / "model.pt"

    train(_train_args(data_file, output_path=output_file))
    output = capsys.readouterr().out
    assert "Saved best model" in output
    assert "Model saved" in output


def test_train_no_output_path(tmp_path, capsys):
    """When args.output is None, no file is saved."""
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file)

    train(_train_args(data_file, output_path=None))
    output = capsys.readouterr().out
    assert "Model saved" not in output


def test_main_runs_training(tmp_path, monkeypatch):
    """main() parses args and runs training."""
    data_file = tmp_path / "train.bin"
    _make_training_data(data_file, num_positions=20)
    output_file = tmp_path / "weights.pt"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_nnue.py",
            "--data",
            str(data_file),
            "--output",
            str(output_file),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--patience",
            "10",
        ],
    )
    main()
    assert output_file.exists()
