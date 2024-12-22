import copy

class ChessBot:
    def __init__(self):
        self.board = self.initialize_board()
        self.turn = 'white'
        self.castling_rights = {'white': {'kingside': True, 'queenside': True},
                                'black': {'kingside': True, 'queenside': True}}
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
        """
        Prints the current state of the chessboard with rank and file labels on all sides, ensuring proper alignment.
        """
        # File labels (columns) on the top
        print("    a  b  c  d  e  f  g  h")  
        
        # Top border with file labels
        print("  +-------------------------+")
        
        for row in range(8):
            rank = 8 - row  # Ranks are numbered from 8 to 1, from top to bottom
            row_str = f"{rank} |"  # Add rank label on the left side of each row
            for col in range(8):
                piece = self.board[row][col]
                row_str += f" {piece} "  # Add the piece (or empty space) to the row with extra spacing
            row_str += f"| {rank}"  # Add rank label on the right side of the row
            print(row_str)  # Print the row with rank labels on both sides
        
        # Bottom border with file labels
        print("  +-------------------------+")
        
        # File labels (columns) on the bottom
        print("    a  b  c  d  e  f  g  h")  

    def is_within_bounds(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def is_under_attack(self, row, col, opponent_turn):
        opponent_moves = self.generate_all_moves(opponent_turn, validate_check=False)
        return any(move[2] == (row, col) for move in opponent_moves)
    
    def get_piece_moves(self, piece, row, col):
        moves = []
        directions = {
            'P': [(-1, 0)],  # White pawn moves forward
            'p': [(1, 0)],   # Black pawn moves forward
            'R': [(0, 1), (0, -1), (1, 0), (-1, 0)],  # Rook directions
            'N': [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)],  # Knight moves
            'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # Bishop directions
            'Q': [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (0, -1), (1, 0), (-1, 0)],  # Queen (rook + bishop)
            'K': [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]   # King moves
        }

        if piece == 'P':  # White pawn
            # Single forward move
            if self.is_within_bounds(row - 1, col) and self.board[row - 1][col] == '.':
                moves.append((row, col, row - 1, col))
                # Double forward move (only from the starting position)
                if row == 6 and self.board[row - 2][col] == '.':
                    moves.append((row, col, row - 2, col))

            # Diagonal captures
            for dr, dc in [(-1, -1), (-1, 1)]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c].islower():  # Capture opponent piece
                        moves.append((row, col, r, c))
                    elif (r, c) == self.en_passant_target:  # En passant
                        moves.append((row, col, r, c))

        elif piece == 'p':  # Black pawn
            # Single forward move
            if self.is_within_bounds(row + 1, col) and self.board[row + 1][col] == '.':
                moves.append((row, col, row + 1, col))
                # Double forward move (only from the starting position)
                if row == 1 and self.board[row + 2][col] == '.':
                    moves.append((row, col, row + 2, col))

            # Diagonal captures
            for dr, dc in [(1, -1), (1, 1)]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c].isupper():  # Capture opponent piece
                        moves.append((row, col, r, c))
                    elif (r, c) == self.en_passant_target:  # En passant
                        moves.append((row, col, r, c))


        # For sliding pieces (rook, bishop, queen)
        elif piece in 'RrBbQq':
            for dr, dc in directions[piece.upper()]:
                r, c = row + dr, col + dc
                while self.is_within_bounds(r, c):
                    if self.board[r][c] == '.':
                        moves.append((row, col, r, c))
                    elif (piece.isupper() and self.board[r][c].islower()) or (piece.islower() and self.board[r][c].isupper()):
                        moves.append((row, col, r, c))
                        break
                    else:  # Stop if the square is occupied by a same-side piece
                        break
                    r, c = r + dr, c + dc

        # For knights and kings
        elif piece in 'NnKk':
            for dr, dc in directions[piece.upper()]:
                r, c = row + dr, col + dc
                if self.is_within_bounds(r, c):
                    if self.board[r][c] == '.' or (piece.isupper() and self.board[r][c].islower()) or (piece.islower() and self.board[r][c].isupper()):
                        moves.append((row, col, r, c))

        return moves



    def generate_all_moves(self, turn, validate_check=True):
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (turn == 'white' and piece.isupper()) or (turn == 'black' and piece.islower()):
                    piece_moves = self.get_piece_moves(piece, row, col)
                    for move in piece_moves:
                        if not self.is_within_bounds(move[2], move[3]):  # Check bounds
                            continue
                        if validate_check and not self.does_move_leave_king_safe(move):
                            continue
                        moves.append(move)
        return moves



    def does_move_leave_king_safe(self, move):
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]
        board_copy = copy.deepcopy(self.board)
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece
        king_pos = self.king_positions[self.turn]
        if piece in 'Kk':  # Update king position if it moves
            king_pos = (end_row, end_col)
        opponent_turn = 'black' if self.turn == 'white' else 'white'
        is_safe = not self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)
        self.board = board_copy
        return is_safe

    def evaluate_position(self):
        score = 0
        for row in self.board:
            for piece in row:
                score += self.piece_values.get(piece, 0)
        return score if self.turn == 'white' else -score

    def bot_move(self, bot_side):
        """
        Makes the bot's move for the assigned side (white or black).
        """
        moves = self.generate_all_moves(bot_side)
        best_move = None
        best_score = float('-inf') if bot_side == 'white' else float('inf')

        for move in moves:
            start_row, start_col, end_row, end_col = move
            board_copy = copy.deepcopy(self.board)
            self.board[start_row][start_col] = '.'
            self.board[end_row][end_col] = board_copy[start_row][start_col]
            score = self.evaluate_position()
            if (bot_side == 'white' and score > best_score) or (bot_side == 'black' and score < best_score):
                best_score = score
                best_move = move
            self.board = board_copy  # Undo move

        if best_move:
            self.make_move(best_move)
            print(f"Bot plays: {self.convert_to_algebraic(best_move)}")

    def make_move(self, move):
        """
        Makes a move on the board and updates the game state.
        """
        start_row, start_col, end_row, end_col = move
        piece = self.board[start_row][start_col]

        # Ensure the starting square contains a valid piece
        if piece == '.':
            raise ValueError("Invalid move: No piece to move from the starting square.")

        # Move the piece
        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = piece

        # Update the king's position if moved
        if piece == 'K':
            self.king_positions['white'] = (end_row, end_col)
        elif piece == 'k':
            self.king_positions['black'] = (end_row, end_col)

        # Handle en passant capture
        if self.en_passant_target and piece in ('P', 'p'):
            if (end_row, end_col) == self.en_passant_target:
                if piece == 'P':  # White pawn
                    self.board[end_row + 1][end_col] = '.'
                elif piece == 'p':  # Black pawn
                    self.board[end_row - 1][end_col] = '.'

        # Handle pawn promotion
        if piece == 'P' and end_row == 0:  # White pawn promotion
            self.board[end_row][end_col] = 'Q'  # Promote to Queen
        elif piece == 'p' and end_row == 7:  # Black pawn promotion
            self.board[end_row][end_col] = 'q'  # Promote to Queen

        # Update en passant target
        if piece in ('P', 'p') and abs(start_row - end_row) == 2:
            self.en_passant_target = ((start_row + end_row) // 2, start_col)
        else:
            self.en_passant_target = None

        # Reset castling rights if rooks or king move
        if piece == 'K':
            self.castling_rights['white']['kingside'] = False
            self.castling_rights['white']['queenside'] = False
        elif piece == 'k':
            self.castling_rights['black']['kingside'] = False
            self.castling_rights['black']['queenside'] = False
        elif piece == 'R':
            if start_row == 7 and start_col == 0:  # Queenside rook
                self.castling_rights['white']['queenside'] = False
            elif start_row == 7 and start_col == 7:  # Kingside rook
                self.castling_rights['white']['kingside'] = False
        elif piece == 'r':
            if start_row == 0 and start_col == 0:  # Queenside rook
                self.castling_rights['black']['queenside'] = False
            elif start_row == 0 and start_col == 7:  # Kingside rook
                self.castling_rights['black']['kingside'] = False


    def parse_move(self, move_str):
        """
        Parses a move in algebraic notation (e.g., 'e2 e4') into internal board coordinates.
        """
        try:
            start, end = move_str.strip().split()
            start_row = 8 - int(start[1])  # Convert rank (1-8) to row (7-0)
            start_col = ord(start[0].lower()) - ord('a')  # Convert file (a-h) to column (0-7)
            end_row = 8 - int(end[1])  # Convert rank (1-8) to row (7-0)
            end_col = ord(end[0].lower()) - ord('a')  # Convert file (a-h) to column (0-7)
            if self.is_within_bounds(start_row, start_col) and self.is_within_bounds(end_row, end_col):
                return (start_row, start_col, end_row, end_col)
            else:
                return None
        except Exception:
            return None

    def convert_to_algebraic(self, move):
        start_row, start_col, end_row, end_col = move
        start = f"{chr(start_col + ord('a'))}{8 - start_row}"
        end = f"{chr(end_col + ord('a'))}{8 - end_row}"
        return f"{start} {end}"

    def is_in_check(self, turn):
        king_pos = self.king_positions[turn]
        opponent_turn = 'black' if turn == 'white' else 'white'
        return self.is_under_attack(king_pos[0], king_pos[1], opponent_turn)

    def is_checkmate(self):
        """Check if the current player is in checkmate."""
        if self.is_in_check(self.turn) and not self.generate_all_moves(self.turn):
            return True
        return False


    def is_stalemate(self):
        if not self.is_in_check(self.turn) and not self.generate_all_moves(self.turn):
            return True
        return False

    def play(self):
        """
        Main game loop: alternates between user and bot moves, ensuring valid inputs and game progression.
        """
        # Allow the user to choose their side
        while True:
            user_side = input("Do you want to play as 'white' or 'black'? ").strip().lower()
            if user_side in ['white', 'black']:
                bot_side = 'black' if user_side == 'white' else 'white'
                print(f"You are playing as {user_side}. The bot will play as {bot_side}.")
                break
            else:
                print("Invalid choice. Please type 'white' or 'black'.")

        self.turn = 'white'  # White always starts in chess
        self.print_board()

        while True:
            if self.is_checkmate():
                print(f"Checkmate! {self.turn} loses.")
                print(f"{'White' if self.turn == 'black' else 'Black'} wins!")
                break
            if self.is_stalemate():
                print("Stalemate! It's a draw.")
                break

            if self.turn == user_side:  # Player's turn
                move_str = input("Enter your move (e.g., 'e2 e4') or type 'resign' to quit: ").strip()
                if move_str.lower() == 'resign':
                    print(f"{user_side.capitalize()} resigns. {bot_side.capitalize()} wins!")
                    break
                parsed_move = self.parse_move(move_str)
                if parsed_move in self.generate_all_moves(user_side):
                    self.make_move(parsed_move)
                else:
                    print("Invalid move. Try again.")
                    continue
            else:  # Bot's turn
                print("Bot is thinking...")
                self.bot_move(bot_side)

            self.turn = 'black' if self.turn == 'white' else 'white'
            self.print_board()