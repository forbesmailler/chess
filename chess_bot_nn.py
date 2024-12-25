import copy
import pickle
import numpy as np
from utils import fen_to_binary_features

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """

    def __init__(self, game, model_path, scaler_path, side='black'):
        self.game = game
        self.side = side
        self.start_time = None
        self.debug = True

        # Load the neural network model

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def choose_move(self):
        # Generate legal moves
        moves = self.game.generate_all_moves(self.side, validate_check=True)
        if not moves:
            return

        if self.debug:
            print(f"[DEBUG] {self.side} has {len(moves)} moves")

        if len(moves) == 1:
            best_move = moves[0]
            self._make_and_print_move(best_move)
            return

        # Use neural network to evaluate moves
        best_move = self.evaluate_moves_with_nn(moves)
        if best_move:
            self._make_and_print_move(best_move)

    def evaluate_moves_with_nn(self, moves):
        """
        Use a neural network to evaluate the moves and select the best one.
        """
        best_move = None
        best_eval = float('-inf')

        for move in moves:

            # Save game state
            board_copy = copy.deepcopy(self.game.board)
            turn_save = self.game.turn

            # Make the move
            self.game.make_move(move)

            # Convert the board to NN input format
            nn_input = fen_to_binary_features(self.game.get_fen())

            # Predict the value using the neural network
            eval_value = np.sum(self.model.predict_proba(self.scaler.transform(nn_input.reshape(1, -1)))  * np.array([-1, 0, 1]))

            if self.debug:
                print(f"[DEBUG] Move {move}, eval={eval_value:.2f}")

            # Update best move
            if eval_value > best_eval:
                best_eval = eval_value
                best_move = move

            # Restore game state
            self.restore_game_state(board_copy, turn_save)

        return best_move

    def _make_and_print_move(self, move):
        # Move is either a normal move or a castling move
        self.game.make_move(move)
        print(f"Bot plays: {self.game.convert_to_algebraic(move)}")

    def restore_game_state(self, board_copy, turn_save):
        self.game.board = board_copy
        self.game.turn = turn_save
