# lichess.py
import logging
import torch
import berserk
import chess
from berserk.exceptions import ResponseError, ApiError
from train_bot import ChessNet, MCTS

logging.basicConfig(level=logging.INFO)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = ChessNet().to(DEVICE)
model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
model.eval()

# Initialize MCTS
mcts = MCTS(model, sims=200, device=DEVICE)

# Lichess client setup
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)

MY_ID = client.account.get()['id']


def _extract_player_id(p):
    if not isinstance(p, dict):
        return None
    user = p.get('user')
    if isinstance(user, dict) and 'id' in user:
        return user['id']
    if 'id' in p:
        return p['id']
    return p.get('name')


def _try_make_move(game_id: str, uci: str) -> bool:
    """
    Attempt to send a move. Returns True on success, False on any error.
    """
    try:
        client.bots.make_move(game_id, uci)
        logging.info(f"Game {game_id}: played move {uci}")
        return True
    except ResponseError as e:
        logging.warning(f"Game {game_id}: move {uci} rejected ({e})")
    except ApiError as e:
        logging.warning(f"Game {game_id}: API error on move {uci} ({e})")
    except Exception as e:
        logging.error(f"Game {game_id}: unexpected error on move {uci} ({e})")
    return False


def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    try:
        first = next(stream)
    except StopIteration:
        logging.error(f"No events for game {game_id}")
        return

    if first.get('type') != 'gameFull':
        logging.error(f"Expected gameFull, got {first.get('type')} in {game_id}")
        return

    # extract white/black IDs
    if 'players' in first:
        raw_w = first['players'].get('white')
        raw_b = first['players'].get('black')
    else:
        raw_w = first.get('white')
        raw_b = first.get('black')

    white_id = _extract_player_id(raw_w)
    black_id = _extract_player_id(raw_b)
    our_white = (white_id == MY_ID)

    logging.info(
        f"Game {game_id}: our ID={MY_ID}, "
        f"white_id={white_id}, black_id={black_id}, "
        f"playing as {'White' if our_white else 'Black'}"
    )

    # initial position
    init_moves = first.get('state', {}).get('moves', '')
    board = chess.Board()
    for uci in init_moves.split():
        board.push(chess.Move.from_uci(uci))

    # if it's our turn at the start, play immediately
    if (board.turn == chess.WHITE and our_white) or \
       (board.turn == chess.BLACK and not our_white):
        root = mcts.search(board)
        if root.children:
            best = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best.uci()):
                return  # bail out on failure

    # process updates
    for event in stream:
        if event.get('type') != 'gameState':
            continue

        # if game ended, stop
        if event.get('status') != 'started':
            logging.info(f"Game {game_id} ended: status={event.get('status')}")
            break

        # rebuild board
        moves = event.get('moves', '')
        board = chess.Board()
        for uci in moves.split():
            board.push(chess.Move.from_uci(uci))

        # if it's our turn, play
        if (board.turn == chess.WHITE and our_white) or \
           (board.turn == chess.BLACK and not our_white):

            root = mcts.search(board)
            if not root.children:
                logging.warning(f"Game {game_id}: no legal moves")
                continue

            best = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best.uci()):
                break  # bail out on error


def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])


if __name__ == '__main__':
    main()
