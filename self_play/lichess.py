# lichess_bot.py (fixed imports)
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
session = berserk.TokenSession(open('token.txt', 'r').read().strip())
client = berserk.Client(session=session)

MY_ID = client.account.get()['id']

def handle_game(game_id: str):
    # open the game state stream and parse initial gameFull event
    stream = client.bots.stream_game_state(game_id)
    try:
        first = next(stream)
    except StopIteration:
        logging.error(f"No events received for game {game_id}")
        return
    if first.get('type') != 'gameFull':
        logging.error(f"Expected gameFull event, got {first.get('type')} for game {game_id}")
        return
    # extract white/black IDs
    if 'players' in first:
        players = first['players']
        white_id = players['white']['user']['id']
        black_id = players['black']['user']['id']
    else:
        white_id = first.get('white')
        black_id = first.get('black')
    our_white = (white_id == MY_ID)
    logging.info(f"Game {game_id}: our ID={MY_ID}, white_id={white_id}, black_id={black_id}, playing as {'White' if our_white else 'Black'}")

    # process subsequent gameState events
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        moves = event.get('moves', '')
        board = chess.Board()
        for uci in moves.split():
            board.push(chess.Move.from_uci(uci))
        # if game over, exit
        if board.is_game_over():
            logging.info(f"Game {game_id} over with result {board.result()}")
            break
        # check if it's our turn
        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
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
