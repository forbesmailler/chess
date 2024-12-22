import copy

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
        This includes pawn moves (with en passant), rook/bishop/queen sliding, knight, king.
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
            # Single push
            if self.is_within_bounds(row - 1, col) and self.board[row - 1][col] == '.':
                moves.append((row, col, row - 1, col))
                # Double push from row 6
                if row == 6 and self.board[row - 2][col] == '.':
                    moves.append((row, col, row - 2, col))
            # Captures (+ possible en passant)
            for dr, dc in [(-1, -1), (-1, 1)]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c].islower():
                        moves.append((row, col, r, c))
                    elif (r, c) == self.en_passant_target:
                        moves.append((row, col, r, c))

        # Black Pawn
        elif piece == 'p':
            # Single push
            if self.is_within_bounds(row + 1, col) and self.board[row + 1][col] == '.':
                moves.append((row, col, row + 1, col))
                # Double push from row 1
                if row == 1 and self.board[row + 2][col] == '.':
                    moves.append((row, col, row + 2, col))
            # Captures (+ possible en passant)
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
        return moves

    def does_move_leave_king_safe(self, move):
        """
        Check if executing 'move' will leave the current player's king in check.
        Temporarily apply the move, check for check, then revert the move.
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

        # Revert to original board
        self.board = board_copy
        return is_safe

    def evaluate_position(self):
        """
        Basic material-only evaluation.
        Sum up piece values based on self.piece_values:
          - White pieces contribute positively,
          - Black pieces contribute negatively.
        """
        score = 0
        for row in self.board:
            for piece in row:
                score += self.piece_values.get(piece, 0)
        return score if self.turn == 'white' else -score

    def bot_move(self, bot_side):
        """Bot chooses and makes a move for its side (white or black)."""
        moves = self.generate_all_moves(bot_side, validate_check=True)
        if not moves:
            return

        best_move = None
        best_score = float('-inf') if bot_side == 'white' else float('inf')

        # Attempt to find a checkmate move first (optional).
        for move in moves:
            if self.is_checkmate():
                best_move = move
                break

        # Otherwise, pick a move based on evaluation.
        if not best_move:
            for move in moves:
                start_row, start_col, end_row, end_col = move
                board_copy = copy.deepcopy(self.board)

                # Simulate the move
                self.board[start_row][start_col] = '.'
                self.board[end_row][end_col] = board_copy[start_row][start_col]
                score = self.evaluate_position()

                if bot_side == 'white' and score > best_score:
                    best_score = score
                    best_move = move
                elif bot_side == 'black' and score < best_score:
                    best_score = score
                    best_move = move

                # Revert the board
                self.board = board_copy

        if best_move:
            self.make_move(best_move)
            print(f"Bot plays: {self.convert_to_algebraic(best_move)}")

    def make_move(self, move):
        """
        Execute the given move and update the game state.
        This method also handles castling, en passant, and promotions.
        """
        # If castling move is detected, handle separately
        if isinstance(move, tuple) and len(move) == 3 and move[0] == "castle":
            _, turn, side = move
            self.execute_castling(turn, side)
            return

        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]

        if piece == '.':
            raise ValueError("Invalid move: No piece to move.")

        # Move the piece
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Update king position if a king has moved
        if piece == 'K':
            self.king_positions['white'] = (end_row, end_col)
        elif piece == 'k':
            self.king_positions['black'] = (end_row, end_col)

        # Handle en passant
        if self.en_passant_target and piece in ('P', 'p'):
            if (end_row, end_col) == self.en_passant_target:
                if piece == 'P':
                    self.board[end_row + 1][end_col] = '.'
                else:
                    self.board[end_row - 1][end_col] = '.'

        # Handle promotion (white or black pawns reaching back rank)
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

        # Update castling rights if relevant pieces moved
        if piece == 'K':
            self.castling_rights['white']['kingside'] = False
            self.castling_rights['white']['queenside'] = False
        elif piece == 'k':
            self.castling_rights['black']['kingside'] = False
            self.castling_rights['black']['queenside'] = False
        elif piece == 'R':
            if start_row == 7 and start_col == 0:
                self.castling_rights['white']['queenside'] = False
            elif start_row == 7 and start_col == 7:
                self.castling_rights['white']['kingside'] = False
        elif piece == 'r':
            if start_row == 0 and start_col == 0:
                self.castling_rights['black']['queenside'] = False
            elif start_row == 0 and start_col == 7:
                self.castling_rights['black']['kingside'] = False

    def execute_castling(self, turn, side):
        """Physically execute castling for the given side (kingside or queenside)."""
        if turn == 'white':
            row = 7
            # Kingside
            if side == 'kingside':
                self.board[row][4] = '.'
                self.board[row][6] = 'K'
                self.board[row][7] = '.'
                self.board[row][5] = 'R'
                self.king_positions['white'] = (7, 6)
                self.castling_rights['white']['kingside'] = False
                self.castling_rights['white']['queenside'] = False
            # Queenside
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
            # Kingside
            if side == 'kingside':
                self.board[row][4] = '.'
                self.board[row][6] = 'k'
                self.board[row][7] = '.'
                self.board[row][5] = 'r'
                self.king_positions['black'] = (0, 6)
                self.castling_rights['black']['kingside'] = False
                self.castling_rights['black']['queenside'] = False
            # Queenside
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
        Returns None if invalid or illegal.
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

        # Parse normal moves like "e2 e4"
        try:
            start, end = move_str.strip().split()
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

    def is_castling_legal(self, turn, side):
        """
        Check the basic castling conditions:
          - King and rook haven't moved.
          - No pieces in between.
          - King not in check.
          - King does not pass through or end up on a square under attack.
        """
        if not self.castling_rights[turn][side]:
            return False

        row = 7 if turn == 'white' else 0
        king_pos_required = (row, 4)
        opponent_turn = 'black' if turn == 'white' else 'white'

        # Ensure king is on e1 (white) or e8 (black)
        if self.king_positions[turn] != king_pos_required:
            return False

        # If king is currently in check, can't castle
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
        Convert move tuples back to standard algebraic notation (e.g., (6,4,4,4) -> 'e2 e4').
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
        """Return True if the current player is in checkmate."""
        if self.is_in_check(self.turn):
            moves = self.generate_all_moves(self.turn, validate_check=True)
            return len(moves) == 0
        return False

    def is_stalemate(self):
        """Return True if the current player is stalemated."""
        return not self.is_in_check(self.turn) and not self.generate_all_moves(self.turn)

    def play(self):
        """
        Main game loop: user chooses a side, then we alternate between user and bot.
        """
        while True:
            user_side = input("Do you want to play as 'white' or 'black'? ").strip().lower()
            if user_side in ['white', 'black']:
                bot_side = 'black' if user_side == 'white' else 'white'
                print(f"You are playing as {user_side}. The bot plays as {bot_side}.")
                break
            print("Invalid choice. Please type 'white' or 'black'.")

        self.turn = 'white'
        self.print_board()

        while True:
            if self.is_checkmate():
                print(f"Checkmate! {'White' if self.turn == 'black' else 'Black'} wins.")
                break
            if self.is_stalemate():
                print("Stalemate! It's a draw.")
                break

            if self.turn == user_side:
                move_str = input("Enter your move (e.g., 'e2 e4', 'O-O', or 'O-O-O') or 'resign': ").strip()
                if move_str.lower() == 'resign':
                    print(f"{user_side.capitalize()} resigns. {bot_side.capitalize()} wins!")
                    break

                parsed_move = self.parse_move(move_str)
                user_moves = self.generate_all_moves(user_side)

                # Check if user attempt is castling or in the list of valid moves
                if (isinstance(parsed_move, tuple) and parsed_move and parsed_move[0] == "castle"):
                    self.make_move(parsed_move)
                elif parsed_move in user_moves:
                    self.make_move(parsed_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            else:
                self.bot_move(bot_side)

            # Switch turn
            self.turn = 'black' if self.turn == 'white' else 'white'
            self.print_board()


if __name__ == "__main__":
    bot = ChessBot()
    bot.play()
