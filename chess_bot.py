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
        """Initialize a standard 8x8 chessboard."""
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
        """Print the current chessboard state with rank/file labels."""
        print("    a  b  c  d  e  f  g  h")
        print("  +-------------------------+")
        for row in range(8):
            rank = 8 - row
            row_str = f"{rank} |"
            for col in range(8):
                row_str += f" {self.board[row][col]} "
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
        Generate opponent's *pseudo-legal* moves (validate_check=False).
        """
        opponent_moves = self.generate_all_moves(opponent_turn, validate_check=False)
        return any((r2 == row and c2 == col) for (_, _, r2, c2) in opponent_moves)

    def get_piece_moves(self, piece, row, col):
        """Generate all pseudo-legal moves for a given piece on (row, col)."""
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

        # Sliding pieces: R, B, Q (and their lowercase counterparts)
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
                    if (self.board[r][c] == '.') or \
                       (piece.isupper() and self.board[r][c].islower()) or \
                       (piece.islower() and self.board[r][c].isupper()):
                        moves.append((row, col, r, c))

        return moves

    def generate_all_moves(self, turn, validate_check=True):
        """
        Return all moves for 'turn'. If validate_check=True,
        only return moves that do not leave own king in check.
        """
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (turn == 'white' and piece.isupper()) or (turn == 'black' and piece.islower()):
                    pseudo_moves = self.get_piece_moves(piece, row, col)
                    for move in pseudo_moves:
                        # Skip out of bounds
                        if not self.is_within_bounds(move[2], move[3]):
                            continue
                        # Validate check
                        if validate_check and not self.does_move_leave_king_safe(move):
                            continue
                        moves.append(move)
        return moves

    def does_move_leave_king_safe(self, move):
        """Check if executing 'move' leaves the current player's king in check."""
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]
        board_copy = copy.deepcopy(self.board)

        # Make the move
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # If it's the king moving, update temporary king position
        king_pos = self.king_positions[self.turn]
        if piece in 'Kk':
            king_pos = (end_row, end_col)

        # Check if king is under attack
        opponent_turn = 'black' if self.turn == 'white' else 'white'
        is_safe = not self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

        # Undo move
        self.board = board_copy
        return is_safe

    def evaluate_position(self):
        """
        Basic evaluation: sum of piece values.
        White pieces add points; black pieces subtract points.
        """
        score = 0
        for row in self.board:
            for piece in row:
                score += self.piece_values.get(piece, 0)
        return score if self.turn == 'white' else -score

    def bot_move(self, bot_side):
        """Bot chooses a move for its side (white or black)."""
        moves = self.generate_all_moves(bot_side, validate_check=True)
        if not moves:
            return

        best_move = None
        best_score = float('-inf') if bot_side == 'white' else float('inf')

        # (Optional) Attempt to find a checkmate move
        for move in moves:
            if self.is_checkmate():
                best_move = move
                break

        # Otherwise, pick move based on evaluation
        if not best_move:
            for move in moves:
                start_row, start_col, end_row, end_col = move
                board_copy = copy.deepcopy(self.board)

                # Simulate move
                self.board[start_row][start_col] = '.'
                self.board[end_row][end_col] = board_copy[start_row][start_col]
                score = self.evaluate_position()

                # Choose best or worst score depending on side
                if (bot_side == 'white' and score > best_score) or \
                   (bot_side == 'black' and score < best_score):
                    best_score = score
                    best_move = move

                self.board = board_copy

        # Execute chosen move
        if best_move:
            self.make_move(best_move)
            print(f"Bot plays: {self.convert_to_algebraic(best_move)}")

    def make_move(self, move):
        """Execute the given move and update the game state."""
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]

        if piece == '.':
            raise ValueError("Invalid move: No piece to move.")

        # Move piece
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Update king positions if moved
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

        # Pawn promotion
        if piece == 'P' and end_row == 0:
            self.board[end_row][end_col] = 'Q'
        elif piece == 'p' and end_row == 7:
            self.board[end_row][end_col] = 'q'

        # Update en passant target
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
            if start_row == 7 and start_col == 0:
                self.castling_rights['white']['queenside'] = False
            elif start_row == 7 and start_col == 7:
                self.castling_rights['white']['kingside'] = False
        elif piece == 'r':
            if start_row == 0 and start_col == 0:
                self.castling_rights['black']['queenside'] = False
            elif start_row == 0 and start_col == 7:
                self.castling_rights['black']['kingside'] = False

    def parse_move(self, move_str):
        """Parse algebraic notation (e.g., 'e2 e4') into board coordinates."""
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

    def convert_to_algebraic(self, move):
        """Convert board coordinates back to standard algebraic notation."""
        sr, sc, er, ec = move
        start = f"{chr(sc + ord('a'))}{8 - sr}"
        end = f"{chr(ec + ord('a'))}{8 - er}"
        return f"{start} {end}"

    def is_in_check(self, turn):
        """Check if the current player's king is in check."""
        king_pos = self.king_positions[turn]
        opponent_turn = 'black' if turn == 'white' else 'white'
        return self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

    def is_checkmate(self):
        """Check if the current player is checkmated."""
        if self.is_in_check(self.turn):
            moves = self.generate_all_moves(self.turn, validate_check=True)
            if not moves:
                return True
        return False

    def is_stalemate(self):
        """Check if the game is in stalemate."""
        return not self.is_in_check(self.turn) and not self.generate_all_moves(self.turn)

    def play(self):
        """
        Main game loop: user chooses side, then alternate turns between user and bot.
        """
        while True:
            user_side = input("Do you want to play as 'white' or 'black'? ").strip().lower()
            if user_side in ['white', 'black']:
                bot_side = 'black' if user_side == 'white' else 'white'
                print(f"You are playing as {user_side}. The bot plays as {bot_side}.")
                break
            else:
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
                move_str = input("Enter your move (e.g., 'e2 e4') or 'resign': ").strip()
                if move_str.lower() == 'resign':
                    print(f"{user_side.capitalize()} resigns. {bot_side.capitalize()} wins!")
                    break
                parsed_move = self.parse_move(move_str)
                user_moves = self.generate_all_moves(user_side)
                if parsed_move in user_moves:
                    self.make_move(parsed_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            else:
                self.bot_move(bot_side)

            self.turn = 'black' if self.turn == 'white' else 'white'
            self.print_board()

if __name__ == "__main__":
    bot = ChessBot()
    bot.play()
