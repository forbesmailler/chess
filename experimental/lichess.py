import argparse
import logging
import threading
import time
from functools import lru_cache

import berserk
import chess
import joblib
import numpy as np
from berserk.exceptions import ApiError, ResponseError

# Constants
WIN_VALUE = 1
LOGISTIC_MODEL_PATH = "chess_lr.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    piece_arr = np.zeros(12 * 64, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        piece_arr[idx * 64 + sq] = 1.0

    # Castling rights features (4)
    castling = np.array(
        [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ],
        dtype=np.float32,
    )

    base = np.concatenate([piece_arr, castling])  # length = 768 + 4 = 772

    n_pieces = len(board.piece_map())
    factor = (n_pieces - 2) / 30

    # Multiply at the end
    return np.concatenate([base * factor, base * (1.0 - factor)])


class SimpleEngine:
    def __init__(self, model):
        self.model = model

    def evaluate(self, board: chess.Board) -> float:
        # Terminal detection
        if board.is_checkmate():
            # If it's white's turn, white is checkmated -> black won
            return -WIN_VALUE if board.turn == chess.WHITE else WIN_VALUE
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0.0
        # Probability-based evaluation
        features = extract_features(board.fen())
        proba = self.model.predict_proba([features])[0]
        return proba[2] - proba[0]

    @lru_cache(maxsize=100000)
    def negamax(self, board_fen: str, depth: int, alpha: float, beta: float) -> float:
        board = chess.Board(board_fen)
        if depth == 0 or board.is_game_over():
            val = self.evaluate(board)
            return val if board.turn == chess.WHITE else -val

        value = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board.fen(), depth - 1, -beta, -alpha)
            board.pop()
            if score > value:
                value = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return value

    def get_best_move(self, board: chess.Board) -> chess.Move:
        depth = 3
        best_move = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board.fen(), depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move


# API client setup
def setup_client(token: str) -> berserk.Client:
    session = berserk.TokenSession(token)
    return berserk.Client(session=session)


# Attempt a move


def _try_make_move(game_id: str, uci: str) -> bool:
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            client.bots.make_move(game_id, uci)
            if attempt > 1:
                logger.info(f"Game {game_id}: succeeded on attempt {attempt}")
            return True
        except (ResponseError, ApiError) as e:
            logger.warning(
                f"Attempt {attempt}/{max_attempts} – Game {game_id}: could not play move {uci}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Attempt {attempt}/{max_attempts} – Game {game_id}: unexpected error on move {uci}: {e}"
            )
        time.sleep(1)  # wait a second before retrying
    logger.error(
        f"Game {game_id}: failed to play move {uci} after {max_attempts} attempts"
    )
    return False


# Helper to play best move


def play_best_move(game_id: str, board: chess.Board) -> bool:
    move = engine.get_best_move(board)
    if move and _try_make_move(game_id, move.uci()):
        board.push(move)
        return True
    return False


# Handle a single game stream
def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    first = next(stream, None)
    if not first or first.get("type") != "gameFull":
        logger.error(f"No gameFull for {game_id}")
        return

    our_white = first.get("white", {}).get("id") == MY_ID
    logger.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    board = chess.Board()
    moves = first.get("state", {}).get("moves", "").split()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    ply_count = len(moves)

    # Initial move if it's our turn
    if (board.turn == chess.WHITE and our_white) or (
        board.turn == chess.BLACK and not our_white
    ):
        logger.info(
            f"Eval after ply {ply_count} (white-persp): {engine.evaluate(board):.4f}"
        )
        if play_best_move(game_id, board):
            ply_count += 1

    # Main loop
    for event in stream:
        if event.get("type") != "gameState" or event.get("status") != "started":
            break

        new_moves = event.get("moves", "").split()
        for uci in new_moves[ply_count:]:
            actor = (
                "Bot"
                if (
                    (ply_count % 2 == 0 and our_white)
                    or (ply_count % 2 == 1 and not our_white)
                )
                else "Opponent"
            )
            logger.info(f"Game {game_id}: {actor} played move {uci}")
            board.push(chess.Move.from_uci(uci))
            ply_count += 1

        logger.info(
            f"Eval after ply {ply_count} (white-persp): {engine.evaluate(board):.4f}"
        )
        if (board.turn == chess.WHITE and our_white) or (
            board.turn == chess.BLACK and not our_white
        ):
            if play_best_move(game_id, board):
                ply_count += 1


# Entry point
def main(token: str):
    global client, MY_ID, engine
    client = setup_client(token)
    MY_ID = client.account.get()["id"]
    engine = SimpleEngine(model=joblib.load(LOGISTIC_MODEL_PATH))

    for ev in client.bots.stream_incoming_events():
        if ev["type"] == "challenge":
            client.bots.accept_challenge(ev["challenge"]["id"])
        elif ev["type"] == "gameStart":
            game_id = ev["game"]["id"]
            threading.Thread(target=handle_game, args=(game_id,), daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess chess bot")
    parser.add_argument("--token", required=True, help="Lichess API token")
    args = parser.parse_args()
    main(args.token)
