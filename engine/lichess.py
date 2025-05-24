import os
import logging
import random
import numpy as np
import chess
import berserk
import joblib
from functools import lru_cache
from berserk.exceptions import ResponseError, ApiError

SEARCH_DEPTH = 3
LOGISTIC_MODEL_PATH = 'chess_lr.joblib'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    arr = np.zeros(12 * 64, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        arr[ch * 64 + sq] = 1.0
    n_pieces = len(board.piece_map())
    factor1 = (n_pieces - 2) / 30
    factor2 = (32 - n_pieces) / 30
    return np.concatenate([arr * factor1, arr * factor2], axis=0)

class SimpleEngine:
    def __init__(self, depth=SEARCH_DEPTH, model=None):
        self.max_depth = depth
        self.model = model

    def evaluate(self, board: chess.Board) -> float:
        features = extract_features(board.fen())
        proba = self.model.predict_proba([features])[0]
        val = proba[2] - proba[0]
        return val

    @lru_cache(maxsize=100000)
    def negamax(self, board_fen: str, depth: int, alpha: float, beta: float) -> float:
        board = chess.Board(board_fen)
        if depth == 0 or board.is_game_over():
            val = self.evaluate(board)
            return val if board.turn == chess.WHITE else -val

        value = -float('inf')
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
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board.fen(), self.max_depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        return best_move

def setup_client():
    with open('token.txt') as f:
        token = f.read().strip()
    session = berserk.TokenSession(token)
    return berserk.Client(session=session)

client = setup_client()
MY_ID = client.account.get()['id']

engine = SimpleEngine(model=joblib.load(LOGISTIC_MODEL_PATH))

def _try_make_move(game_id: str, uci: str) -> bool:
    try:
        client.bots.make_move(game_id, uci)
        return True
    except (ResponseError, ApiError) as e:
        logger.warning(f"Game {game_id}: could not play move {uci}: {e}")
    except Exception as e:
        logger.error(f"Game {game_id}: unexpected error on move {uci}: {e}")
    return False

def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    first = next(stream, None)
    if not first or first.get('type') != 'gameFull':
        logger.error(f"No gameFull for {game_id}")
        return

    our_white = (first.get('white', {}).get('id') == MY_ID)
    logger.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    board = chess.Board()
    moves = first.get('state', {}).get('moves', '').split()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    ply_count = len(moves)

    # Initial move if it's our turn
    if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
        evaluation = engine.evaluate(board)
        logger.info(f"Eval after ply {ply_count} (white-persp): {evaluation:.4f}")
        best_move = engine.get_best_move(board)
        if best_move and _try_make_move(game_id, best_move.uci()):
            board.push(best_move)
            ply_count += 1

    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            break

        new_moves = event.get('moves', '').split()
        for uci in new_moves[ply_count:]:
            actor = "Bot" if ((ply_count % 2 == 0 and our_white) or (ply_count % 2 == 1 and not our_white)) else "Opponent"
            logger.info(f"Game {game_id}: {actor} played move {uci}")
            board.push(chess.Move.from_uci(uci))
            ply_count += 1

        evaluation = engine.evaluate(board)
        logger.info(f"Eval after ply {ply_count} (white-persp): {evaluation:.4f}")

        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            best_move = engine.get_best_move(board)
            if best_move:
                _try_make_move(game_id, best_move.uci())
                board.push(best_move)
                ply_count += 1

def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])

if __name__ == '__main__':
    main()
