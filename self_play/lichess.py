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
def setup_client():
    with open('token.txt', 'r') as f:
        token = f.read().strip()
    session = berserk.TokenSession(token)
    return berserk.Client(session=session)

client = setup_client()
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

    our_white = (first.get('white', {}).get('id') == MY_ID)
    logger.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    # pick our color for MCTS
    player_color = chess.WHITE if our_white else chess.BLACK
    mcts = MCTS(model, player_color=player_color, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE)

    # Initialize board and history
    moves = first.get('state', {}).get('moves', '').split()
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    ply_count = len(moves)
    root = None
    examples = []

    # If it's our turn to move at the start
    if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        # always log white-perspective
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {ply_count} (white-persp): {adj_val:.4f}")

        sims = max(50, int(MCTS_SIMS * (1 - ply_count / 200)))
        mcts.sims = sims
        root = mcts.search(board)

        examples.append(board.fen())
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        _try_make_move(game_id, best_move.uci())
        board.push(best_move)
        ply_count += 1

        # reuse subtree
        root = root.children.get(best_move)
        if root:
            root.parent = None

    # process incoming moves
    result = None
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break

        new_moves = event.get('moves', '').split()
        for uci in new_moves[ply_count:]:
            actor = "Bot" if ((ply_count % 2 == 0 and our_white) or (ply_count % 2 == 1 and not our_white)) else "Opponent"
            logger.info(f"Game {game_id}: {actor} played move {uci}")
            board.push(chess.Move.from_uci(uci))

            if root and chess.Move.from_uci(uci) in root.children:
                root = root.children[chess.Move.from_uci(uci)]
                root.parent = None
            else:
                root = None
            ply_count += 1

        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {ply_count} (white-persp): {adj_val:.4f}")

        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            sims = max(50, int(MCTS_SIMS * (1 - ply_count / 200)))
            mcts.sims = sims
            root = mcts.search(board, root)
            examples.append(board.fen())

            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
            board.push(best_move)
            ply_count += 1
            root = root.children.get(best_move)
            if root:
                root.parent = None

    # Training phase
    base_z = 1.0 if result == 'white' else -1.0 if result == 'black' else 0.0
    batch = [(fen, base_z * ((-1) ** i)) for i, fen in enumerate(examples)]
    if batch:
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Training on {len(batch)} examples: loss={loss:.4f}")

        # Calibrate final bias so initial position evaluates to zero
        init_board = chess.Board()
        feat0 = state_to_tensor(init_board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw0 = model(feat0).cpu().item()
        pre_act = torch.atanh(torch.tensor(raw0, device=DEVICE))
        model.fc3.bias.data -= pre_act
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
