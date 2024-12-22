import copy
from collections import defaultdict
import time

class ChessBot:
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

        # For tracking threefold repetition
        self.position_counts = defaultdict(int)

        # For the fifty-move rule: halfmove_clock is the number of halfmoves
        # since the last capture or pawn advance.
        self.halfmove_clock = 0

        # Store which side the user is playing (set in play())
        self.user_side = None

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
            'N': [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)],
            'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            'Q': [(-1, -1), (-1, 1), (1, -1), (1, 1),
                  (0, 1), (0, -1), (1, 0), (-1, 0)],
            'K': [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
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
                    elif (piece.isupper() and self.board[r][c].islower()) or \
                         (piece.islower() and self.board[r][c].isupper()):
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
                    if self.board[r][c] == '.' or \
                       (piece.isupper() and self.board[r][c].islower()) or \
                       (piece.islower() and self.board[r][c].isupper()):
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
                if (turn == 'white' and piece.isupper()) or (turn == 'black' and piece.islower()):
                    pseudo_moves = self.get_piece_moves(piece, row, col)
                    for move in pseudo_moves:
                        if not self.is_within_bounds(move[2], move[3]):
                            continue
                        if validate_check and not self.does_move_leave_king_safe(move):
                            continue
                        moves.append(move)

        # Only add castling moves if validate_check is True, to avoid recursion in evaluate_position
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

    def evaluate_position(self):
        """
        IMPROVED EVALUATION FUNCTION:
        1. Check immediate checkmate/stalemate (highest priority).
        2. Material sum (existing logic).
        3. Mobility.
        """
        # --- 1. CHECKMATE/STALEMATE CHECKS ---
        # We'll do a quick local check. If it's the current player's turn:
        if self.is_checkmate():
            # The side to move is checkmated => huge negative from their perspective
            if self.turn == 'white':
                return -9999  # White checkmated => losing
            else:
                return 9999   # Black checkmated => from White's perspective, big positive => so from Black's perspective, big negative => we invert
        if self.is_stalemate():
            # Stalemate => it's a draw => 0
            return 0

        # --- 2. MATERIAL ---
        material_score = 0
        for row in self.board:
            for piece in row:
                material_score += self.piece_values.get(piece, 0)

        # --- 3. MOBILITY ---
        white_moves = self.generate_all_moves('white', validate_check=False)
        black_moves = self.generate_all_moves('black', validate_check=False)
        mobility_score = (len(white_moves) - len(black_moves)) * 0.1

        total_score = material_score + mobility_score

        # If black to move, invert sign, so a "positive" White advantage becomes negative if it's black's viewpoint
        if self.turn == 'black':
            total_score = -total_score

        return total_score

    def bot_move(self, bot_side):
        """Bot chooses and makes a move for its side (white or black)."""
        moves = self.generate_all_moves(bot_side, validate_check=True)
        if not moves:
            return

        best_move = None
        best_score = float('-inf') if bot_side == 'white' else float('inf')

        # Pick the move based on the improved evaluate_position()
        for move in moves:
            start_row, start_col, end_row, end_col = move
            piece = self.board[start_row][start_col]

            board_copy = copy.deepcopy(self.board)

            # Handle potential promotions by testing all promotion pieces
            if (piece == 'P' and end_row == 0) or (piece == 'p' and end_row == 7):
                if piece == 'P':
                    promo_list = ['Q', 'R', 'B', 'N']
                else:
                    promo_list = ['q', 'r', 'b', 'n']

                best_promo_eval = float('-inf') if bot_side == 'white' else float('inf')
                best_promo_piece = promo_list[0]

                for promo in promo_list:
                    self.board[start_row][start_col] = '.'
                    self.board[end_row][end_col] = promo
                    promo_score = self.evaluate_position()
                    self.board = copy.deepcopy(board_copy)

                    if bot_side == 'white' and promo_score > best_promo_eval:
                        best_promo_eval = promo_score
                        best_promo_piece = promo
                    elif bot_side == 'black' and promo_score < best_promo_eval:
                        best_promo_eval = promo_score
                        best_promo_piece = promo

                # Place the best promotion piece, evaluate
                self.board = copy.deepcopy(board_copy)
                self.board[start_row][start_col] = '.'
                self.board[end_row][end_col] = best_promo_piece
                final_score = self.evaluate_position()

            else:
                # Normal move
                self.board[start_row][start_col] = '.'
                self.board[end_row][end_col] = piece
                final_score = self.evaluate_position()

            self.board = copy.deepcopy(board_copy)

            # Track best move
            if bot_side == 'white' and final_score > best_score:
                best_score = final_score
                best_move = move
            elif bot_side == 'black' and final_score < best_score:
                best_score = final_score
                best_move = move

        # Execute the best move
        if best_move:
            sr, sc, er, ec = best_move
            piece = self.board[sr][sc]

            # Double-check if it's a promotion
            if (piece == 'P' and er == 0) or (piece == 'p' and er == 7):
                if piece == 'P':
                    promo_list = ['Q', 'R', 'B', 'N']
                else:
                    promo_list = ['q', 'r', 'b', 'n']

                board_before = copy.deepcopy(self.board)
                best_promo_eval = float('-inf') if bot_side == 'white' else float('inf')
                best_promo_piece = promo_list[0]

                for promo in promo_list:
                    self.board[sr][sc] = '.'
                    self.board[er][ec] = promo
                    test_eval = self.evaluate_position()
                    self.board = copy.deepcopy(board_before)

                    if bot_side == 'white' and test_eval > best_promo_eval:
                        best_promo_eval = test_eval
                        best_promo_piece = promo
                    elif bot_side == 'black' and test_eval < best_promo_eval:
                        best_promo_eval = test_eval
                        best_promo_piece = promo

                # Manually finalize
                self.board = copy.deepcopy(board_before)
                self.board[sr][sc] = '.'
                self.board[er][ec] = best_promo_piece
                if piece.lower() == 'p':
                    self.halfmove_clock = 0
                else:
                    self.halfmove_clock += 1
                self.en_passant_target = None
                self.record_position()

                print(f"Bot promotes to {best_promo_piece}!")
                print(f"Bot plays: {self.convert_to_algebraic(best_move)}")
            else:
                self.make_move(best_move)
                print(f"Bot plays: {self.convert_to_algebraic(best_move)}")

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

        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        captured_piece = None
        if board_copy_check(self, start_row, start_col, end_row, end_col):
            captured_piece = piece

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

        # User promotion logic (bot promotions handled in bot_move)
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
            if self.is_within_bounds(start_row, start_col) and \
               self.is_within_bounds(end_row, end_col):
                return (start_row, start_col, end_row, end_col)
        except:
            pass

        return None

    def record_position(self):
        """
        Record the current position in position_counts for threefold repetition.
        The position key includes board layout, turn, castling rights, and en passant.
        """
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
            castling_info.append(f"{side[0]}K{rights['kingside']}Q{rights['queenside']}")

        ep_str = str(self.en_passant_target) if self.en_passant_target else '-'
        turn_str = self.turn[0]  # 'w' or 'b'

        return f"{board_str} {turn_str} {' '.join(castling_info)} {ep_str}"

    def check_draw_conditions(self):
        """
        Check:
          1) Threefold repetition
          2) Fifty-move rule
          3) Insufficient material
        """
        # 1) Threefold repetition
        pos_key = self.create_position_key()
        if self.position_counts[pos_key] >= 3:
            print("Draw by threefold repetition!")
            return True

        # 2) Fifty-move rule
        if self.halfmove_clock >= 50:
            print("Draw by fifty-move rule!")
            return True

        # 3) Insufficient material
        if self.is_insufficient_material():
            print("Draw by insufficient material!")
            return True

        return False

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

        # Check for King + single minor
        minor_white = set(['K', 'N', 'B'])
        minor_black = set(['k', 'n', 'b'])

        if (all(p in minor_white for p in white_pieces)
                and all(p in minor_black for p in black_pieces)):
            if len(white_pieces) <= 2 and len(black_pieces) <= 2:
                return True

        return False

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

        if self.king_positions[turn] != king_pos_required:
            return False
        if self.is_in_check(turn):
            return False

        if side == 'kingside':
            if self.board[row][5] != '.' or self.board[row][6] != '.':
                return False
            if self.is_under_attack(row, 5, opponent_turn) or \
               self.is_under_attack(row, 6, opponent_turn):
                return False
        else:
            if (self.board[row][1] != '.' or
                self.board[row][2] != '.' or
                self.board[row][3] != '.'):
                return False
            if self.is_under_attack(row, 3, opponent_turn) or \
               self.is_under_attack(row, 2, opponent_turn):
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
        """True if the 'turn' player's king is in check."""
        king_pos = self.king_positions[turn]
        opponent_turn = 'black' if turn == 'white' else 'white'
        return self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

    def is_checkmate(self):
        """True if current player is in checkmate."""
        if self.is_in_check(self.turn):
            moves = self.generate_all_moves(self.turn, validate_check=True)
            return len(moves) == 0
        return False

    def is_stalemate(self):
        """True if current player is stalemated."""
        return not self.is_in_check(self.turn) and not self.generate_all_moves(self.turn)

    def play(self):
        """
        Main game loop: user picks side, alternate turns between user and bot.
        Check for draws each move: threefold repetition, 50-move rule,
        insufficient material, etc.
        """
        while True:
            user_side = input("Do you want to play as 'white' or 'black'? ").strip().lower()
            if user_side in ['white', 'black']:
                self.user_side = user_side
                bot_side = 'black' if user_side == 'white' else 'white'
                print(f"You are playing as {user_side}. The bot plays as {bot_side}.")
                break
            print("Invalid choice. Please type 'white' or 'black'.")

        self.turn = 'white'
        self.record_position()
        self.print_board()

        while True:
            if self.check_draw_conditions():
                break
            if self.is_checkmate():
                print(f"Checkmate! {'White' if self.turn == 'black' else 'Black'} wins.")
                break
            if self.is_stalemate():
                print("Stalemate! It's a draw.")
                break

            if self.turn == self.user_side:
                move_str = input("Enter your move (e.g., 'e2 e4', 'O-O', or 'O-O-O') or 'resign': ").strip()
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
                self.bot_move(bot_side)

            self.turn = 'black' if self.turn == 'white' else 'white'
            self.print_board()

        time.sleep(5) # optional pause at game end


def board_copy_check(chessbot, sr, sc, er, ec):
    """
    Returns True if (er, ec) was occupied prior to the move,
    so that we know if a capture occurred.
    """
    return chessbot.board[er][ec] != '.'


if __name__ == "__main__":
    bot = ChessBot()
    bot.play()
