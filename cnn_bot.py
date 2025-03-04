from copy import deepcopy
import numpy as np
import torch
from cnn_model import ChessCNN  # Ensure this imports your model class definition
from game import ChessGame
from utils import fen_to_binary_features

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """
    def __init__(self, game: ChessGame, model_path, side='black', max_depth=3):
        self.game = game
        self.side = side
        self.max_depth = max_depth
        self.debug = True

        # Load the neural network model from the .pth file.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessCNN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.eval()

    def choose_move(self):
        moves = self.game.generate_all_moves(self.side, validate_check=True)
        if not moves:
            return

        if self.debug:
            print(f"[DEBUG] {self.side} has {len(moves)} moves")

        if len(moves) == 1:
            best_move = moves[0]
            self._make_and_print_move(best_move)
            return

        best_move = self.alpha_beta_root(moves)
        if best_move:
            self._make_and_print_move(best_move)

    def alpha_beta_root(self, moves):
        best_move = None
        best_eval = float('-inf')
        for move in moves:
            eval = self.alpha_beta(move, depth=self.max_depth - 1,
                                     alpha=float('-inf'), beta=float('inf'),
                                     maximizing=False)
            if eval > best_eval:
                best_eval = eval
                best_move = move

            if self.debug:
                print(f"[DEBUG] Move {move}, eval={eval:.4f}")

        return best_move

    def alpha_beta(self, move, depth, alpha, beta, maximizing):
        # Save game state
        board_copy = deepcopy(self.game.board)
        turn_save = deepcopy(self.game.turn)
        king_positions_save = deepcopy(self.game.king_positions)
        castling_rights_save = deepcopy(self.game.castling_rights)
        en_passant_save = deepcopy(self.game.en_passant_target)
        halfmove_clock_save = deepcopy(self.game.halfmove_clock)

        # Make the move
        self.game.make_move(move)
        
        if (depth == 0 or self.game.is_checkmate() or self.game.check_draw_conditions()):
            eval = self.evaluate_position()
        else:
            next_side = self.game.turn
            next_moves = self.game.generate_all_moves(next_side, validate_check=True)
            if not next_moves:
                eval = self.evaluate_position()
            else:
                if maximizing:
                    eval = float('-inf')
                    for nxt_move in next_moves:
                        current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, False)
                        eval = max(eval, current_eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                else:
                    eval = float('inf')
                    for nxt_move in next_moves:
                        current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, True)
                        eval = min(eval, current_eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
        # Restore game state
        self.restore_game_state(board_copy,
                                turn_save,
                                king_positions_save,
                                castling_rights_save,
                                en_passant_save,
                                halfmove_clock_save)
        return eval

    def evaluate_position(self):
        if self.game.is_checkmate():
            return -9999 if self.game.turn == 'white' else 9999
        if self.game.check_draw_conditions():
            print("draw")
            return 0

        nn_input = fen_to_binary_features(self.game.get_fen()).reshape(1, -1)
        device = next(self.model.parameters()).device
        nn_input_tensor = torch.tensor(nn_input, dtype=torch.float32, device=device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(nn_input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            # Debug prints:
            # print(f"Logits: {logits.cpu().numpy()}")
            # print(f"Probabilities: {probabilities.cpu().numpy()}")
            score = (probabilities[0, 2] - probabilities[0, 0]).item()
        return score


    def _make_and_print_move(self, move):
        self.game.make_move(move)
        print(f"Bot plays: {self.game.convert_to_algebraic(move)}")

    def restore_game_state(self, board_copy,
                           turn_save,
                           king_positions_save,
                           castling_rights_save,
                           en_passant_save,
                           halfmove_clock_save):
        self.game.board = board_copy
        self.game.turn = turn_save
        self.game.king_positions = king_positions_save
        self.game.castling_rights = castling_rights_save
        self.game.en_passant_target = en_passant_save
        self.game.halfmove_clock = halfmove_clock_save
