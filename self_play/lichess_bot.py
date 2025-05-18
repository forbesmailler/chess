# lichess_bot.py
import os
import berserk
import chess
import torch
from network import ChessNet
from mcts import MCTS, state_to_tensor, UCI_TO_IDX, ALL_UCIS

# load
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessNet().to(DEVICE)
model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
model.eval()
mcts = MCTS(model, sims=200, device=DEVICE)

# Lichess client
with open('token.txt', 'r', encoding='utf-8') as f:
    token = f.read()

session = berserk.TokenSession(token)
client  = berserk.Client(session=session)

def handle_game(game_id: str):
    """
    Stream game state and make moves when it's our turn.
    """
    for event in client.bots.stream_game_state(game_id):
        if event['type'] == 'gameFull':
            # initial full game state if needed
            continue
        elif event['type'] == 'gameState':
            moves = event['moves'].split()
            board = chess.Board()
            for mv in moves:
                board.push(chess.Move.from_uci(mv))

            # Determine if it's our turn
            # event['white'] is the white player ID
            my_id = client.account.get()['id']
            our_color_is_white = (event['white'] == my_id)
            if board.turn == chess.WHITE and our_color_is_white:
                is_our_turn = True
            elif board.turn == chess.BLACK and not our_color_is_white:
                is_our_turn = True
            else:
                is_our_turn = False

            if is_our_turn:
                # Run MCTS to get best move
                root = mcts.search(board)
                best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
                client.bots.make_move(game_id, best_move.uci())


def main():
    """
    Main loop: accept challenges and start handling games.
    """
    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            # Extract the nested challenge ID
            challenge_id = event['challenge']['id']
            client.bots.accept_challenge(challenge_id)
        elif event['type'] == 'gameStart':
            # Extract the nested game ID
            game_id = event['game']['id']
            handle_game(game_id)


if __name__ == '__main__':
    main()
