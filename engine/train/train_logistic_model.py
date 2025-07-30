from tqdm import tqdm
import pandas as pd
import chess
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import os
import joblib

def extract_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    piece_arr = np.zeros(12 * 64, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        piece_arr[idx * 64 + sq] = 1.0

    # Castling rights features (4)
    castling = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)

    base = np.concatenate([piece_arr, castling])  # length = 768 + 4 = 772

    n_pieces = len(board.piece_map())
    factor = (n_pieces - 2) / 30

    # Multiply at the end
    return np.concatenate([base * factor, base * (1.0 - factor)])

def process_dataset(df: pd.DataFrame, size: int, desc: str):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:size]
    X = np.zeros((len(df), 2 * (12 * 64 + 4)), dtype=np.float32)
    y = df['value'].values.astype(np.float32)
    for i, fen in tqdm(enumerate(df['FEN']), total=len(df), desc=f"Featurizing {desc}"):
        X[i] = extract_features(fen)
    return X, y

def load_split_save(train_csv: str,
                    val_csv: str,
                    train_size: int = 1_000_000,
                    val_size: int = 100_000):

    if os.path.exists("X_train.parquet") and os.path.exists("y_train.parquet") \
       and os.path.exists("X_val.parquet") and os.path.exists("y_val.parquet"):
        print("Loading pre-saved Parquet datasets...")
        X_train = pd.read_parquet("X_train.parquet").values
        y_train = pd.read_parquet("y_train.parquet")["value"].values
        X_val   = pd.read_parquet("X_val.parquet").values
        y_val   = pd.read_parquet("y_val.parquet")["value"].values
        return X_train, y_train, X_val, y_val

    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    X_train, y_train = process_dataset(df_train, train_size, 'train')
    X_val,   y_val   = process_dataset(df_val,   val_size,   'val')

    pd.DataFrame(X_train).to_parquet("X_train.parquet", index=False)
    pd.DataFrame({"value": y_train}).to_parquet("y_train.parquet", index=False)
    pd.DataFrame(X_val).to_parquet("X_val.parquet", index=False)
    pd.DataFrame({"value": y_val}).to_parquet("y_val.parquet", index=False)

    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_split_save(
        'train.csv', 'val.csv',
        train_size=1_000_000, val_size=100_000
    )