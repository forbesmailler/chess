# --- lichess.py (updated) ---
import os
import logging
import math

import torch
import chess
import berserk
from berserk.exceptions import ResponseError, ApiError
from train_bot import (
    ChessNet,
    MCTS,
    state_to_tensor,
    DEVICE,
    LR,
    MCTS_SIMS,
    train_on_batch
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChessNet().to(DEVICE)
if os.path.exists('best.pth'):
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    logger.info("Loaded existing best.pth")
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()['id']


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

    raw_w = first.get('white')
    our_white = (raw_w.get('id') == MY_ID)
    logger.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    # Reconstruct initial board
    initial_moves = first.get('state', {}).get('moves', '').split()
    board = chess.Board()
    for uci in initial_moves:
        board.push(chess.Move.from_uci(uci))

    examples = []
    prev_move_count = len(initial_moves)

    # Log eval of initial position before first move if it's our turn
    if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {prev_move_count} (white-persp): {adj_val:.4f}")

        root = MCTS(model, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE).search(board)
        examples.append(board.fen())
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        _try_make_move(game_id, best_move.uci())
        prev_move_count += 1

    result = None

    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break

        # Get full move list
        moves = event.get('moves', '').split()
        # Log any new moves (bot or opponent)
        for i in range(prev_move_count, len(moves)):
            uci = moves[i]
            is_our_move = (i % 2 == 0 and our_white) or (i % 2 == 1 and not our_white)
            actor = "Bot" if is_our_move else "Opponent"
            logger.info(f"Game {game_id}: {actor} played move {uci}")
        prev_move_count = len(moves)

        # Rebuild board after moves
        board = chess.Board()
        for uci in moves:
            board.push(chess.Move.from_uci(uci))

        # Log evaluation after every ply
        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {len(moves)} (white-persp): {adj_val:.4f}")

        # Only play on our turns
        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            root = MCTS(model, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE).search(board)
            examples.append(board.fen())
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break

    # Training phase
    base_z = 1.0 if result == 'white' else -1.0 if result == 'black' else 0.0
    batch = [(fen, base_z * ((-1) ** i)) for i, fen in enumerate(examples)]

    if batch:
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Training on {len(batch)} examples: loss={loss:.4f}")
        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved best.pth")


def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])


if __name__ == '__main__':
    main()