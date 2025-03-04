import copy
import time
from collections import defaultdict

from copy import deepcopy
import numpy as np
import torch
from chess_game import ChessGame
from utils import fen_to_binary_features

import torch
import torch.nn as nn
import torch.nn.functional as F

def board_copy_check(game, sr, sc, er, ec):
    """
    Returns True if (er, ec) was occupied prior to the move,
    so that we know if a capture occurred.
    """
    return game.board[er][ec] != '.'

class ChessGame:
    def __init__(self):
        self.board = self.initialize_board()
        self.turn = 'white'
        self.castling_rights = {
            'white': {'kingside': True, 'queenside': True},
            'black': {'kingside': True, 'queenside': True}
        }
        self.en_passant_target = None
        self.king_positions = {'white': (7, 4), 'black': (0, 4)}
        self.piece_values = {
            'P': 1, 'p': -1,
            'N': 3, 'n': -3,
            'B': 3, 'b': -3,
            'R': 5, 'r': -5,
            'Q': 9, 'q': -9,
            'K': 0, 'k': 0
        }

        self.move_count = 0

        # For tracking threefold repetition
        self.position_counts = defaultdict(int)

        # For the fifty-move rule: halfmove_clock is the number of halfmoves
        # since the last capture or pawn advance.
        self.halfmove_clock = 0

        # Store which side the user is playing (set in play())
        self.user_side = None

        # Add a flag to indicate when we're in search mode.
        self.searching = False

    def initialize_board(self):
        """
        Initialize a standard 8x8 chessboard.
        Uppercase = White pieces; lowercase = Black pieces.
        """
        return [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]

    def print_board(self):
        """Print the current chessboard with rank/file labels."""
        print("    a  b  c  d  e  f  g  h")
        print("  +-------------------------+")
        for row_index in range(8):
            rank = 8 - row_index
            row_str = f"{rank} |"
            for col_index in range(8):
                row_str += f" {self.board[row_index][col_index]} "
            row_str += f"| {rank}"
            print(row_str)
        print("  +-------------------------+")
        print("    a  b  c  d  e  f  g  h")

    def is_within_bounds(self, row, col):
        """Check if (row, col) is within the board."""
        return 0 <= row < 8 and 0 <= col < 8

    def is_under_attack(self, row, col, opponent_turn):
        """
        Check if (row, col) is under attack by the opponent.
        Generate the opponent's pseudo-legal moves (validate_check=False).
        """
        opponent_moves = self.generate_all_moves(opponent_turn, validate_check=False)
        return any((r2 == row and c2 == col) for (_, _, r2, c2) in opponent_moves)

    def get_piece_moves(self, piece, row, col):
        """
        Generate all pseudo-legal moves for a given piece on (row, col).
        This includes pawn moves (with en passant), rook/bishop/queen sliding,
        knight, and king moves.
        """
        moves = []
        directions = {
            'P': [(-1, 0)],  # White pawn
            'p': [(1, 0)],   # Black pawn
            'R': [(0, 1), (0, -1), (1, 0), (-1, 0)],
            'N': [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ],
            'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            'Q': [
                (-1, -1), (-1, 1), (1, -1), (1, 1),
                (0, 1), (0, -1), (1, 0), (-1, 0)
            ],
            'K': [
                (-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)
            ]
        }

        # White Pawn
        if piece == 'P':
            if self.is_within_bounds(row - 1, col) and self.board[row - 1][col] == '.':
                moves.append((row, col, row - 1, col))
                if row == 6 and self.board[row - 2][col] == '.':
                    moves.append((row, col, row - 2, col))

            for dr, dc in [(-1, -1), (-1, 1)]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c].islower():
                        moves.append((row, col, r, c))
                    elif (r, c) == self.en_passant_target:
                        moves.append((row, col, r, c))

        # Black Pawn
        elif piece == 'p':
            if self.is_within_bounds(row + 1, col) and self.board[row + 1][col] == '.':
                moves.append((row, col, row + 1, col))
                if row == 1 and self.board[row + 2][col] == '.':
                    moves.append((row, col, row + 2, col))

            for dr, dc in [(1, -1), (1, 1)]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c].isupper():
                        moves.append((row, col, r, c))
                    elif (r, c) == self.en_passant_target:
                        moves.append((row, col, r, c))

        # Sliding pieces: R, B, Q (uppercase or lowercase)
        elif piece in 'RrBbQq':
            for dr, dc in directions[piece.upper()]:
                r, c = row + dr, col + dc
                while self.is_within_bounds(r, c):
                    if self.board[r][c] == '.':
                        moves.append((row, col, r, c))
                    elif (
                        (piece.isupper() and self.board[r][c].islower())
                        or (piece.islower() and self.board[r][c].isupper())
                    ):
                        moves.append((row, col, r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc

        # Knights & Kings
        elif piece in 'NnKk':
            for dr, dc in directions[piece.upper()]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if (
                        self.board[r][c] == '.'
                        or (piece.isupper() and self.board[r][c].islower())
                        or (piece.islower() and self.board[r][c].isupper())
                    ):
                        moves.append((row, col, r, c))

        return moves

    def generate_all_moves(self, turn, validate_check=True):
        """
        Return all moves for 'turn'. If validate_check=True,
        only include moves that do not leave the player's own king in check.
        """
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (turn == 'white' and piece.isupper()) or (
                    turn == 'black' and piece.islower()
                ):
                    pseudo_moves = self.get_piece_moves(piece, row, col)
                    for move in pseudo_moves:
                        if not self.is_within_bounds(move[2], move[3]):
                            continue
                        if validate_check and not self.does_move_leave_king_safe(move):
                            continue
                        moves.append(move)

        # Only add castling moves if validate_check is True, to avoid recursion
        # in evaluate_position
        if validate_check:
            if self.is_castling_legal(turn, 'kingside'):
                moves.append(("castle", turn, "kingside"))
            if self.is_castling_legal(turn, 'queenside'):
                moves.append(("castle", turn, "queenside"))

        return moves

    def does_move_leave_king_safe(self, move):
        """
        Check if executing 'move' leaves the current player's king in check.
        Temporarily apply the move, check for check, then revert.
        """
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]
        board_copy = copy.deepcopy(self.board)

        # Make the move
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Temporarily update king position if a king moved
        king_pos = self.king_positions[self.turn]
        if piece in 'Kk':
            king_pos = (end_row, end_col)

        # Check if king is under attack
        opponent_turn = 'black' if self.turn == 'white' else 'white'
        is_safe = not self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

        # Revert
        self.board = board_copy
        return is_safe

    def make_move(self, move):
        """
        Execute the given move and update the game state.
        (Supports normal moves, castling, en passant, and user promotions.)
        """
        if isinstance(move, tuple) and len(move) == 3 and move[0] == "castle":
            _, turn, side = move
            self.execute_castling(turn, side)
            self.record_position()
            return

        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]

        if piece == '.':
            raise ValueError("Invalid move: No piece to move.")

        captured_piece = None
        if board_copy_check(self, start_row, start_col, end_row, end_col):
            captured_piece = self.board[end_row][end_col]

        # Move the piece
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Update king position if a king has moved
        if piece == 'K':
            self.king_positions['white'] = (end_row, end_col)
        elif piece == 'k':
            self.king_positions['black'] = (end_row, end_col)

        # En passant capture
        if self.en_passant_target and piece in ('P', 'p'):
            if (end_row, end_col) == self.en_passant_target:
                if piece == 'P':
                    self.board[end_row + 1][end_col] = '.'
                    captured_piece = 'p'
                else:
                    self.board[end_row - 1][end_col] = '.'
                    captured_piece = 'P'

        # User promotion logic (bot promotions handled separately)
        if piece == 'P' and end_row == 0:
            if self.turn == self.user_side == 'white':
                promotion_piece = ''
                while promotion_piece not in ['Q', 'R', 'B', 'N']:
                    promotion_piece = input("Promote to (Q, R, B, N)? ").strip().upper()
                self.board[end_row][end_col] = promotion_piece
            else:
                self.board[end_row][end_col] = 'Q'
        elif piece == 'p' and end_row == 7:
            if self.turn == self.user_side == 'black':
                promotion_piece = ''
                while promotion_piece not in ['q', 'r', 'b', 'n']:
                    promotion_piece = input("Promote to (q, r, b, n)? ").strip().lower()
                self.board[end_row][end_col] = promotion_piece
            else:
                self.board[end_row][end_col] = 'q'

        # En passant target update
        if piece in ('P', 'p') and abs(start_row - end_row) == 2:
            self.en_passant_target = ((start_row + end_row) // 2, start_col)
        else:
            self.en_passant_target = None

        # Update castling rights
        if piece == 'K':
            self.castling_rights['white']['kingside'] = False
            self.castling_rights['white']['queenside'] = False
        elif piece == 'k':
            self.castling_rights['black']['kingside'] = False
            self.castling_rights['black']['queenside'] = False
        elif piece == 'R':
            if start_row == 7 and start_col in [0, 7]:
                if start_col == 0:
                    self.castling_rights['white']['queenside'] = False
                else:
                    self.castling_rights['white']['kingside'] = False
        elif piece == 'r':
            if start_row == 0 and start_col in [0, 7]:
                if start_col == 0:
                    self.castling_rights['black']['queenside'] = False
                else:
                    self.castling_rights['black']['kingside'] = False

        # Update halfmove clock for fifty-move rule
        if piece.lower() == 'p' or captured_piece:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.record_position()

    def execute_castling(self, turn, side):
        """Physically execute castling for the given side (kingside or queenside)."""
        if turn == 'white':
            row = 7
            if side == 'kingside':
                self.board[row][4] = '.'
                self.board[row][6] = 'K'
                self.board[row][7] = '.'
                self.board[row][5] = 'R'
                self.king_positions['white'] = (7, 6)
                self.castling_rights['white']['kingside'] = False
                self.castling_rights['white']['queenside'] = False
            else:
                self.board[row][4] = '.'
                self.board[row][2] = 'K'
                self.board[row][0] = '.'
                self.board[row][3] = 'R'
                self.king_positions['white'] = (7, 2)
                self.castling_rights['white']['kingside'] = False
                self.castling_rights['white']['queenside'] = False
        else:
            row = 0
            if side == 'kingside':
                self.board[row][4] = '.'
                self.board[row][6] = 'k'
                self.board[row][7] = '.'
                self.board[row][5] = 'r'
                self.king_positions['black'] = (0, 6)
                self.castling_rights['black']['kingside'] = False
                self.castling_rights['black']['queenside'] = False
            else:
                self.board[row][4] = '.'
                self.board[row][2] = 'k'
                self.board[row][0] = '.'
                self.board[row][3] = 'r'
                self.king_positions['black'] = (0, 2)
                self.castling_rights['black']['kingside'] = False
                self.castling_rights['black']['queenside'] = False

    def parse_move(self, move_str):
        """
        Parse user move (e.g. 'e2 e4', 'O-O', 'O-O-O'), returning a move tuple if valid.
        Returns None if invalid.
        """
        move_str = move_str.strip()

        # Check for castling
        if move_str in ('O-O', 'o-o'):
            if self.is_castling_legal(self.turn, 'kingside'):
                return ("castle", self.turn, "kingside")
            return None
        if move_str in ('O-O-O', 'o-o-o'):
            if self.is_castling_legal(self.turn, 'queenside'):
                return ("castle", self.turn, "queenside")
            return None

        # Parse moves like "e2 e4"
        try:
            start, end = move_str.split()
            start_row = 8 - int(start[1])
            start_col = ord(start[0].lower()) - ord('a')
            end_row = 8 - int(end[1])
            end_col = ord(end[0].lower()) - ord('a')
            if (
                self.is_within_bounds(start_row, start_col)
                and self.is_within_bounds(end_row, end_col)
            ):
                return (start_row, start_col, end_row, end_col)
        except Exception:
            pass

        return None

    def record_position(self):
        """
        Record the current position in position_counts for threefold repetition.
        The position key includes board layout, turn, castling rights, and en passant.
        """
        if self.searching:
            return  # Skip recording during search
        pos_key = self.create_position_key()
        self.position_counts[pos_key] += 1


    def create_position_key(self):
        """
        Create a string that uniquely identifies the position for repetition tracking:
        - board
        - active player
        - castling rights
        - en_passant_target
        """
        rows_joined = [''.join(row) for row in self.board]
        board_str = '/'.join(rows_joined)

        castling_info = []
        for side in ['white', 'black']:
            rights = self.castling_rights[side]
            castling_info.append(
                f"{side[0]}K{rights['kingside']}Q{rights['queenside']}"
            )

        ep_str = str(self.en_passant_target) if self.en_passant_target else '-'
        turn_str = self.turn[0]  # 'w' or 'b'

        return f"{board_str} {turn_str} {' '.join(castling_info)} {ep_str}"

    def is_castling_legal(self, turn, side):
        """
        Check that king/rook haven't moved, no pieces in between,
        king not in check, etc.
        """
        if not self.castling_rights[turn][side]:
            return False

        row = 7 if turn == 'white' else 0
        king_pos_required = (row, 4)
        opponent_turn = 'black' if turn == 'white' else 'white'

        # King must be on the original square
        if self.king_positions[turn] != king_pos_required:
            return False
        # Can't castle if you're currently in check
        if self.is_in_check(turn):
            return False

        if side == 'kingside':
            if self.board[row][5] != '.' or self.board[row][6] != '.':
                return False
            if (
                self.is_under_attack(row, 5, opponent_turn)
                or self.is_under_attack(row, 6, opponent_turn)
            ):
                return False
        else:
            if (
                self.board[row][1] != '.'
                or self.board[row][2] != '.'
                or self.board[row][3] != '.'
            ):
                return False
            if (
                self.is_under_attack(row, 3, opponent_turn)
                or self.is_under_attack(row, 2, opponent_turn)
            ):
                return False

        return True

    def convert_to_algebraic(self, move):
        """
        Convert move tuples back to standard notation. e.g. (6,4,4,4) -> 'e2 e4'.
        For castling, return 'O-O' or 'O-O-O'.
        """
        if isinstance(move, tuple) and move[0] == "castle":
            _, _, side = move
            return "O-O" if side == 'kingside' else "O-O-O"

        sr, sc, er, ec = move
        start = f"{chr(sc + ord('a'))}{8 - sr}"
        end = f"{chr(ec + ord('a'))}{8 - er}"
        return f"{start} {end}"

    def is_in_check(self, turn):
        """Return True if the 'turn' player's king is in check."""
        king_pos = self.king_positions[turn]
        opponent_turn = 'black' if turn == 'white' else 'white'
        return self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

    def is_checkmate(self):
        """True if current player is in checkmate."""
        if self.is_in_check(self.turn):
            moves = self.generate_all_moves(self.turn, validate_check=True)
            return len(moves) == 0
        return False

    def check_draw_conditions(self):
        """Check if any draw condition (stalemate, threefold, 50-move, insufficient)."""
        return (
            self.is_stalemate()
            or self.is_threefold_repetition()
            or self.is_fifty_move_rule()
            or self.is_insufficient_material()
        )

    def is_threefold_repetition(self):
        """Return True if the current position has occurred at least three times."""
        return self.position_counts[self.create_position_key()] >= 3

    def is_fifty_move_rule(self):
        """Return True if there have been at least 50 moves since a capture or pawn move."""
        return self.halfmove_clock >= 100

    def is_insufficient_material(self):
        """
        Check if neither side can force a checkmate based on material:
          - King vs King
          - King + (B or N) vs King
          - King vs King + (B or N)
        """
        white_pieces = []
        black_pieces = []
        for row in self.board:
            for piece in row:
                if piece.isupper():
                    white_pieces.append(piece)
                elif piece.islower():
                    black_pieces.append(piece)

        # King vs King
        if set(white_pieces) == {'K'} and set(black_pieces) == {'k'}:
            return True

        # Check for King + single minor vs. King
        minor_white = {'K', 'N', 'B'}
        minor_black = {'k', 'n', 'b'}

        if (all(p in minor_white for p in white_pieces)
                and all(p in minor_black for p in black_pieces)):
            if len(white_pieces) <= 2 and len(black_pieces) <= 2:
                return True

        return False

    def is_stalemate(self):
        """True if current player is stalemated (not in check, no legal moves)."""
        return not self.is_in_check(self.turn) and not self.generate_all_moves(self.turn)
    
    def get_fen(self):
        """
        Return the current position in standard FEN notation.
        Example of a starting position FEN:
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        # 1) Piece placement
        # Note: row 0 is "top" of our board, which corresponds to the 8th rank in FEN.
        fen_rows = []
        for row in range(8):
            empty_count = 0
            fen_row = ""
            for col in range(8):
                piece = self.board[row][col]
                if piece == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        fen_board = "/".join(fen_rows)

        # 2) Active color
        fen_active_color = 'w' if self.turn == 'white' else 'b'

        # 3) Castling availability
        castling_part = ""
        if self.castling_rights['white']['kingside']:
            castling_part += "K"
        if self.castling_rights['white']['queenside']:
            castling_part += "Q"
        if self.castling_rights['black']['kingside']:
            castling_part += "k"
        if self.castling_rights['black']['queenside']:
            castling_part += "q"
        if castling_part == "":
            castling_part = "-"

        # 4) En passant target
        if self.en_passant_target:
            (r, c) = self.en_passant_target
            # Convert row/col to algebraic, e.g. row=3,col=4 -> 'e5'
            file_char = chr(ord('a') + c)
            rank_num = 8 - r
            fen_en_passant = f"{file_char}{rank_num}"
        else:
            fen_en_passant = "-"

        # 5) Halfmove clock (since last capture or pawn move)
        fen_halfmove = str(self.halfmove_clock)

        # 6) Fullmove number = 1 + (move_count // 2)
        fen_fullmove = str(1 + (self.move_count // 2))

        # Put it all together
        fen_string = f"{fen_board} {fen_active_color} {castling_part} {fen_en_passant} {fen_halfmove} {fen_fullmove}"
        return fen_string


    def play(self, bot):
        """
        Main game loop: user picks side, alternate turns between user and bot.
        Check for draws each move.
        """
        while True:
            user_side = input("Do you want to play as 'white' or 'black'? ").strip().lower()
            if user_side in ['white', 'black']:
                self.user_side = user_side
                bot.side = 'black' if user_side == 'white' else 'white'
                print(f"You are playing as {user_side}. The bot plays as {bot.side}.")
                break
            print("Invalid choice. Please type 'white' or 'black'.")

        self.turn = 'white'
        self.record_position()
        self.print_board()

        while True:
            if self.is_checkmate():
                print(f"Checkmate! {'White' if self.turn == 'black' else 'Black'} wins.")
                break
            if self.is_stalemate():
                print("Stalemate! It's a draw.")
                break
            if self.is_threefold_repetition():
                print("Draw by threefold repetition!")
                break
            if self.is_fifty_move_rule():
                print("Draw by fifty-move rule!")
                break
            if self.is_insufficient_material():
                print("Draw by insufficient material!")
                break

            if self.turn == self.user_side:
                move_str = input(
                    "Enter your move (e.g., 'e2 e4', 'O-O', or 'O-O-O') "
                    "or 'resign': "
                ).strip()
                if move_str.lower() == 'resign':
                    winner = 'white' if self.user_side == 'black' else 'black'
                    print(f"{self.user_side.capitalize()} resigns. {winner.capitalize()} wins!")
                    break

                parsed_move = self.parse_move(move_str)
                user_moves = self.generate_all_moves(self.user_side)

                if isinstance(parsed_move, tuple) and parsed_move and parsed_move[0] == "castle":
                    self.make_move(parsed_move)
                elif parsed_move in user_moves:
                    self.make_move(parsed_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            else:
                # Let the bot choose and play its move
                bot.choose_move()

            # Switch turns
            self.turn = 'black' if self.turn == 'white' else 'white'
            self.move_count += 1
            self.print_board()

        time.sleep(5)  # optional pause at game end

# Configurable hyperparameters
CONV_CHANNELS = [4, 8]         # For faster training you might try fewer channels (e.g. [16, 32])
KERNEL_SIZE = 3                # Common choice is 3, try larger sizes if needed
PADDING = 1                    # With kernel size 3, padding of 1 preserves the spatial dimensions
FC_HIDDEN_LAYERS = [64]        # You can reduce to [256] for a lighter model and faster training

class CNNModel(nn.Module):
    def __init__(self, conv_channels=CONV_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, fc_hidden_layers=FC_HIDDEN_LAYERS):
        super(CNNModel, self).__init__()
        # The board state is 768 features reshaped into (12, 8, 8)
        self.conv_layers = nn.ModuleList()
        in_channels = 12  # 6 piece types * 2 colors
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            in_channels = out_channels
        # Assuming the board remains 8x8 after conv layers (if kernel_size=3 and padding=1)
        conv_output_size = 8 * 8 * conv_channels[-1]
        # Extra 12 features are concatenated
        fc_input_size = conv_output_size + 12

        self.fc_layers = nn.ModuleList()
        for hidden_size in fc_hidden_layers:
            self.fc_layers.append(nn.Linear(fc_input_size, hidden_size))
            fc_input_size = hidden_size
        self.output_layer = nn.Linear(fc_input_size, 3)  # 3 classes: win, draw, loss

    def forward(self, x):
        # x shape: (N, 780)
        board = x[:, :768]   # First 768 features for board state
        extras = x[:, 768:]  # Next 12 features (en passant + castling)
        
        # Reshape board to (N, 12, 8, 8)
        board = board.view(-1, 12, 8, 8)
        for conv in self.conv_layers:
            board = F.relu(conv(board))
        board = board.view(board.size(0), -1)
        
        # Concatenate flattened board features with extra features
        x_combined = torch.cat([board, extras], dim=1)
        for fc in self.fc_layers:
            x_combined = F.relu(fc(x_combined))
        logits = self.output_layer(x_combined)
        return logits

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """
    def __init__(self, game: ChessGame, model_path, side='black', max_depth=2):
        self.game = game
        self.side = side
        self.max_depth = max_depth
        self.debug = True

        # Load the neural network model from the .pth file.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel().to(device)
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
        # Disable recording during the search
        self.game.searching = True

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

        # Re-enable recording after search is done.
        self.game.searching = False

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
