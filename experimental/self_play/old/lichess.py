# lichess_trainable.py
# A Lichess bot that plays, learns from its games, and saves its model after each game.

import logging
import os

import berserk
import chess
import torch
from berserk.exceptions import ApiError, ResponseError
from train_bot import (
    ALL_UCIS,
    DEVICE,
    LR,
    MCTS,
    MCTS_SIMS,
    UCI_TO_IDX,
    ChessNet,
    state_to_tensor,
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)

# load or initialize model
model = ChessNet().to(DEVICE)
if os.path.exists("best.pth"):
    model.load_state_dict(torch.load("best.pth", map_location=DEVICE))
else:
    logging.info("No existing best.pth found. Starting from scratch.")
model.eval()

# optimizer for online training
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open("token.txt", "r") as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()["id"]


# ------------------------- Training utils -------------------------
def tensor_from_pi(pi_dict):
    pi = torch.zeros(len(ALL_UCIS), dtype=torch.float32)
    for uci, p in pi_dict.items():
        pi[UCI_TO_IDX[uci]] = p
    return pi


def train_on_batch(batch):
    st_fens, pis, zs = zip(*batch)
    states = torch.stack([state_to_tensor(chess.Board(fen)) for fen in st_fens]).to(
        DEVICE
    )
    target_pis = torch.stack([tensor_from_pi(pi) for pi in pis]).to(DEVICE)
    target_vals = torch.tensor(zs, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    logits, values = model(states)
    loss = torch.nn.functional.mse_loss(
        values, target_vals
    ) + torch.nn.functional.cross_entropy(logits, target_pis.argmax(dim=1))
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()


# ------------------------- Helper functions -------------------------
def _extract_player_id(p):
    if not isinstance(p, dict):
        return None
    user = p.get("user")
    if isinstance(user, dict) and "id" in user:
        return user["id"]
    if "id" in p:
        return p["id"]
    return p.get("name")


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
    try:
        first = next(stream)
    except StopIteration:
        logging.error(f"No events for game {game_id}")
        return

    if first.get("type") != "gameFull":
        logging.error(f"Expected gameFull, got {first.get('type')}")
        return

    # determine players
    if "players" in first:
        raw_w = first["players"].get("white")
        raw_b = first["players"].get("black")
    else:
        raw_w = first.get("white")
        raw_b = first.get("black")

    white_id = _extract_player_id(raw_w)
    black_id = _extract_player_id(raw_b)
    our_white = white_id == MY_ID

    logging.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    # collect self-play examples
    game_examples = []  # (fen, pi_dict, z)
    board = chess.Board()

    # apply any initial moves
    init_moves = first.get("state", {}).get("moves", "")
    for uci in init_moves.split():
        board.push(chess.Move.from_uci(uci))

    # if it's our turn from the start, play and record
    if (board.turn == chess.WHITE and our_white) or (
        board.turn == chess.BLACK and not our_white
    ):
        root = MCTS(model, sims=MCTS_SIMS, device=DEVICE).search(board)
        counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
        total = sum(counts.values())
        pi = {u: n / total for u, n in counts.items()}
        game_examples.append((board.fen(), pi, None))
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        _try_make_move(game_id, best_move.uci())
        board.push(best_move)

    # play until end
    result = None
    for event in stream:
        if event.get("type") != "gameState":
            continue
        if event.get("status") != "started":
            result = event.get("winner")
            break

        moves = event.get("moves", "")
        board = chess.Board()
        for uci in moves.split():
            board.push(chess.Move.from_uci(uci))

        if (board.turn == chess.WHITE and our_white) or (
            board.turn == chess.BLACK and not our_white
        ):
            root = MCTS(model, sims=MCTS_SIMS, device=DEVICE).search(board)
            counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
            total = sum(counts.values())
            pi = {u: n / total for u, n in counts.items()}
            game_examples.append((board.fen(), pi, None))
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
            board.push(best_move)

    # determine outcome z
    z = 0.0
    if result == "white":
        z = 1.0
    elif result == "black":
        z = -1.0

    # train on all moves from this game
    training_batch = [(fen, pi, z) for fen, pi, _ in game_examples]
    if training_batch:
        loss = train_on_batch(training_batch)
        logging.info(f"Training on {len(training_batch)} examples: loss={loss:.4f}")
        torch.save(model.state_dict(), "best.pth")
        logging.info("Saved best.pth")


# ------------------------- Main loop -------------------------
def main():
    for ev in client.bots.stream_incoming_events():
        if ev["type"] == "challenge":
            client.bots.accept_challenge(ev["challenge"]["id"])
        elif ev["type"] == "gameStart":
            handle_game(ev["game"]["id"])


if __name__ == "__main__":
    main()
