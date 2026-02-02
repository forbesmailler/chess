import os

import chess
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm


def extract_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)

    # 64 * 12 piece square table
    piece_arr = np.zeros(12 * 64, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        piece_arr[idx * 64 + sq] = 1.0

    # Additional features for regular scaling (2)
    # 1. Is white in check?
    # 2. Is black in check?
    white_in_check = 0
    black_in_check = 0
    if board.turn == chess.WHITE:
        white_in_check = 1 if board.is_check() else 0
        board.turn = chess.BLACK
        if not board.is_game_over():
            black_in_check = 1 if board.is_check() else 0
        board.turn = chess.WHITE
    else:
        black_in_check = 1 if board.is_check() else 0
        board.turn = chess.WHITE
        if not board.is_game_over():
            white_in_check = 1 if board.is_check() else 0
        board.turn = chess.BLACK

    n_pieces = len(board.piece_map())

    # Base features for regular scaling
    base_features = np.array([white_in_check, black_in_check], dtype=np.float32)

    # Combine piece array and check features
    base = np.concatenate([piece_arr, base_features])  # length = 768 + 2 = 770

    # Scale by number of pieces factor
    factor = (n_pieces - 2) / 30.0
    scaled_features = np.concatenate(
        [base * factor, base * (1.0 - factor)]
    )  # 770 * 2 = 1540

    # Calculate mobility features separately
    white_mobility = 0.0
    black_mobility = 0.0

    # Count pieces for each color
    white_pieces = sum(
        1 for piece in board.piece_map().values() if piece.color == chess.WHITE
    )
    black_pieces = sum(
        1 for piece in board.piece_map().values() if piece.color == chess.BLACK
    )

    # Only calculate white mobility if white has < 8 pieces
    if white_pieces < 8:
        white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        if board.turn == chess.BLACK:
            board.turn = chess.WHITE
            if not board.is_game_over():
                white_moves = len(list(board.legal_moves))
            board.turn = chess.BLACK

        # Scale by white piece count: max((8 - white_pieces) / 6, 0)
        white_factor = max((8 - white_pieces) / 6.0, 0)
        white_mobility = white_factor * white_moves

    # Only calculate black mobility if black has < 8 pieces
    if black_pieces < 8:
        black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        if board.turn == chess.WHITE:
            board.turn = chess.BLACK
            if not board.is_game_over():
                black_moves = len(list(board.legal_moves))
            board.turn = chess.WHITE

        # Scale by black piece count: max((8 - black_pieces) / 6, 0)
        black_factor = max((8 - black_pieces) / 6.0, 0)
        black_mobility = black_factor * black_moves

    # Return scaled features + mobility features
    return np.concatenate([scaled_features, [white_mobility, black_mobility]])


def process_dataset(df: pd.DataFrame, size: int, desc: str):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:size]
    X = np.zeros(
        (len(df), 2 * (12 * 64 + 2) + 2), dtype=np.float32
    )  # 770 * 2 + 2 = 1542 features
    y = df["value"].values.astype(np.float32)
    for i, fen in tqdm(enumerate(df["FEN"]), total=len(df), desc=f"Featurizing {desc}"):
        X[i] = extract_features(fen)
    return X, y


def load_split_save(
    train_csv: str, val_csv: str, train_size: int = 1_000_000, val_size: int = 100_000
):
    if (
        os.path.exists("X_train.parquet")
        and os.path.exists("y_train.parquet")
        and os.path.exists("X_val.parquet")
        and os.path.exists("y_val.parquet")
    ):
        print("Loading pre-saved Parquet datasets...")
        X_train = pd.read_parquet("X_train.parquet").values
        y_train = pd.read_parquet("y_train.parquet")["value"].values
        X_val = pd.read_parquet("X_val.parquet").values
        y_val = pd.read_parquet("y_val.parquet")["value"].values
        return X_train, y_train, X_val, y_val

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    X_train, y_train = process_dataset(df_train, train_size, "train")
    X_val, y_val = process_dataset(df_val, val_size, "val")

    pd.DataFrame(X_train).to_parquet("X_train.parquet", index=False)
    pd.DataFrame({"value": y_train}).to_parquet("y_train.parquet", index=False)
    pd.DataFrame(X_val).to_parquet("X_val.parquet", index=False)
    pd.DataFrame({"value": y_val}).to_parquet("y_val.parquet", index=False)

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_split_save(
        "train.csv", "val.csv", train_size=1_000_000, val_size=100_000
    )

    print("Fitting logistic regression...")
    lr = LogisticRegression(max_iter=1000, verbose=1)
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_val)
    ll = log_loss(y_val, y_prob)
    print(f"Log loss: {ll:.4f}")

    joblib.dump(lr, "chess_lr.joblib")
