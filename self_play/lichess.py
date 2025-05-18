# lichess.py
import logging
import torch
import berserk
import chess
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

def handle_game(game_id: str):
    # start streaming the game state
    stream = client.bots.stream_game_state(game_id)
    try:
        first = next(stream)
    except StopIteration:
        logging.error(f"No events received for game {game_id}")
        return

    if first.get('type') != 'gameFull':
        logging.error(f"Expected gameFull event, got {first.get('type')} for game {game_id}")
        return

    # extract string IDs for white and black
    if 'players' in first:
        players = first['players']

        wp = players.get('white', {})
        if 'user' in wp and isinstance(wp['user'], dict):
            white_id = wp['user'].get('id')
        else:
            white_id = wp.get('id') or wp.get('name')

        bp = players.get('black', {})
        if 'user' in bp and isinstance(bp['user'], dict):
            black_id = bp['user'].get('id')
        else:
            black_id = bp.get('id') or bp.get('name')
    else:
        # fallback for legacy schema
        white_id = first.get('white')
        black_id = first.get('black')

    our_white = (white_id == MY_ID)
    logging.info(
        f"Game {game_id}: our ID={MY_ID}, "
        f"white_id={white_id}, black_id={black_id}, "
        f"playing as {'White' if our_white else 'Black'}"
    )

    # handle each gameState update
    for event in stream:
        if event.get('type') != 'gameState':
            continue

        moves = event.get('moves', '')
        board = chess.Board()
        for uci in moves.split():
            board.push(chess.Move.from_uci(uci))

        if board.is_game_over():
            logging.info(f"Game {game_id} over: {board.result()}")
            break

        # if it's our turn, play
        if (board.turn == chess.WHITE and our_white) or \
           (board.turn == chess.BLACK and not our_white):

            root = mcts.search(board)
            if not root.children:
                logging.warning(f"Game {game_id}: no legal moves available")
                continue

            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            client.bots.make_move(game_id, best_move.uci())
            logging.info(f"Game {game_id}: played move {best_move}")

def main():
    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.accept_challenge(event['challenge']['id'])
        elif event['type'] == 'gameStart':
            handle_game(event['game']['id'])

if __name__ == '__main__':
    main()
