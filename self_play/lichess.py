# --- lichess.py ---
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
    DEVICE,
    LR,
    MCTS_SIMS,
    train_on_batch
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)
model = ChessNet().to(DEVICE)
if os.path.exists('best.pth'):
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    logging.info("Loaded existing best.pth")
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()['id']

# ------------------------- Helper functions -------------------------
def _try_make_move(game_id: str, uci: str) -> bool:
    try:
        client.bots.make_move(game_id, uci)
        logging.info(f"Game {game_id}: played move {uci}")
        return True
    except (ResponseError, ApiError) as e:
        logging.warning(f"Game {game_id}: could not play move {uci}: {e}")
    except Exception as e:
        logging.error(f"Game {game_id}: unexpected error on move {uci}: {e}")
    return False

# ------------------------- Game handler -------------------------
def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    first = next(stream, None)
    if not first or first.get('type') != 'gameFull':
        logging.error(f"No gameFull for {game_id}")
        return

    raw_w = first.get('white')
    our_white = (raw_w.get('user', {}).get('id') == MY_ID)
    logging.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    board = chess.Board()
    for uci in first.get('state', {}).get('moves', '').split():
        board.push(chess.Move.from_uci(uci))

    examples = []
    result = None
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break
        moves = event.get('moves', '').split()
        board = chess.Board()
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            root = MCTS(model, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE).search(board)
            examples.append(board.fen())
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
    # determine z
    z = 0.0
    if result == 'white':
        z = 1.0
    elif result == 'black':
        z = -1.0

    # train on this game's data
    batch = [(fen, z) for fen in examples]
    if batch:
        loss = train_on_batch(model, optimizer, batch)
        logging.info(f"Training on {len(batch)} examples: loss={loss:.4f}")
        torch.save(model.state_dict(), 'best.pth')
        logging.info("Saved best.pth")

# ------------------------- Main loop -------------------------
def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])

if __name__ == '__main__':
    main()