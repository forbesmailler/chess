import copy
import time

piece_square_tables = {
    'P': [
        [ 0,   0,   0,   0,   0,   0,   0,  0],
        [ 5,   5,   5,  -5,  -5,   5,   5,  5],
        [ 1,   1,   2,   3,   3,   2,   1,  1],
        [ 0.5, 0.5, 1,   2.5, 2.5,  1,   0.5,0.5],
        [ 0,   0,   0,   2,   2,   0,   0,  0],
        [ 0.5,-0.5,-1,   0,   0,  -1,  -0.5,0.5],
        [ 0.5, 1,   1,  -2,  -2,   1,   1,  0.5],
        [ 0,   0,   0,   0,   0,   0,   0,  0],
    ],
    'N': [
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
        [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
        [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
        [-3.0,  0.5, 1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
        [-3.0,  0.0, 1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
        [-3.0,  0.5, 1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
        [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
    ],
    'B': [
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
        [-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
        [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
        [-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
        [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    ],
    'R': [
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [ 0.0,  0.0,  0.0,  0.5,  0.5,  0.0,  0.0,  0.0],
    ],
    'Q': [
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
        [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
        [ 0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
        [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
        [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    ],
    'K': [
        [ 2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0],
        [ 2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0],
        [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
        [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    ],
}

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """

    def __init__(self, game, side='black', max_depth=4, time_limit=30.0):
        """
        :param game: reference to an instance of ChessGame
        :param side: 'white' or 'black'
        :param max_depth: how many plies (levels) deep to search
        :param time_limit: max number of seconds to search
        """
        self.game = game  
        self.side = side
        self.max_depth = max_depth
        self.time_limit = time_limit

        self.start_time = None

        # For debugging, set this to True to see alpha-beta decisions
        self.debug = True

    def choose_move(self):
        """
        Finds the best move for self.side using a deeper alpha-beta
        search (up to self.max_depth). Also includes a time limit to avoid
        taking too long.
        """
        self.start_time = time.time()

        # Generate legal moves for this side
        moves = self.game.generate_all_moves(self.side, validate_check=True)
        if not moves:
            # No moves => checkmate or stalemate
            return

        if self.debug:
            print(f"[DEBUG] {self.side} possible moves: {moves}")

        best_move = None
        if self.side == 'white':
            best_eval = float('-inf')
        else:
            best_eval = float('inf')

        # If there's only 1 legal move, no need for alpha-beta
        if len(moves) == 1:
            best_move = moves[0]
        else:
            best_move, best_eval = self.alpha_beta_root(moves)

        # Execute best move
        if best_move:
            # Handle castling
            if isinstance(best_move, tuple) and best_move[0] == "castle":
                self.game.make_move(best_move)
                print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")
                return

            # Handle promotion or normal move
            sr, sc, er, ec = best_move
            piece = self.game.board[sr][sc]
            if (piece == 'P' and er == 0) or (piece == 'p' and er == 7):
                board_before = copy.deepcopy(self.game.board)
                best_promo_piece, _ = self.choose_best_promotion(
                    sr, sc, er, ec, piece, board_before
                )
                self.game.board = copy.deepcopy(board_before)
                self.game.board[sr][sc] = '.'
                self.game.board[er][ec] = best_promo_piece

                # Update halfmove clock, etc.
                if piece.lower() == 'p':
                    self.game.halfmove_clock = 0
                else:
                    self.game.halfmove_clock += 1
                self.game.en_passant_target = None
                self.game.record_position()
                print(f"Bot promotes to {best_promo_piece}!")

            self.game.make_move(best_move)
            print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")

    # ---------------------- ALPHA-BETA SEARCH ------------------------
    def alpha_beta_root(self, moves):
        best_move = None
        
        if self.side == 'white':
            best_eval = float('-inf')
            for move in moves:
                if self._time_expired():
                    break
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=False)
                if val > best_eval:
                    best_eval = val
                    best_move = move
                if self.debug:
                    print(f"[DEBUG] Root move {move} => eval {val} (white)")

        else:  # self.side == 'black'
            best_eval = float('inf')
            for move in moves:
                if self._time_expired():
                    break
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=True)
                if val < best_eval:
                    best_eval = val
                    best_move = move
                if self.debug:
                    print(f"[DEBUG] Root move {move} => eval {val} (black)")

        return best_move, best_eval

    def alpha_beta(self, move, depth, alpha, beta, maximizing):
        # Save state
        board_copy_state = copy.deepcopy(self.game.board)
        turn_save = self.game.turn
        king_positions_save = self.game.king_positions.copy()
        castling_rights_save = copy.deepcopy(self.game.castling_rights)
        en_passant_save = self.game.en_passant_target
        halfmove_clock_save = self.game.halfmove_clock

        # Make the move
        self.game.make_move(move)

        # If time expired or depth=0 or game is over => evaluate and revert
        if (depth == 0 or self._time_expired()
            or self.game.is_checkmate() or self.game.check_draw_conditions()):
            val = self.evaluate_position()
            self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                    castling_rights_save, en_passant_save, halfmove_clock_save)
            return val

        # Now generate moves for whoever is on move
        next_side = self.game.turn  # 'white' or 'black'
        moves = self.game.generate_all_moves(next_side, validate_check=True)

        if not moves:  
            # No moves => checkmate or stalemate => evaluate
            val = self.evaluate_position()
            self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                    castling_rights_save, en_passant_save, halfmove_clock_save)
            return val

        if maximizing:
            best_eval = float('-inf')
            for nxt_move in moves:
                if self._time_expired():
                    break
                current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, False)
                best_eval = max(best_eval, current_eval)
                alpha = max(alpha, best_eval)
                if beta <= alpha:
                    break
        else:
            best_eval = float('inf')
            for nxt_move in moves:
                if self._time_expired():
                    break
                current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, True)
                best_eval = min(best_eval, current_eval)
                beta = min(beta, best_eval)
                if beta <= alpha:
                    break

        self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                castling_rights_save, en_passant_save, halfmove_clock_save)
        return best_eval

    def _time_expired(self):
        return (time.time() - self.start_time) >= self.time_limit

    # ---------------- EVALUATION ------------------------------------
    def evaluate_position(self):
        # 1. Checkmate / Draw
        if self.game.is_checkmate():
            return -9999 if self.game.turn == 'white' else 9999
        if self.game.check_draw_conditions():
            return 0

        material_score = 0
        bishop_count_white = 0
        bishop_count_black = 0

        # -------------------------------------
        # MAIN LOOP THROUGH BOARD
        # -------------------------------------
        for r in range(8):
            for c in range(8):
                piece = self.game.board[r][c]
                if piece == '.':
                    continue

                # Add piece material value (already positive for White, negative for Black)
                piece_value = self.game.piece_values.get(piece, 0)
                material_score += piece_value

                # piece type (capital letter)
                piece_type = piece.upper()

                # If it's in our piece-square table dict, apply a table bonus/penalty
                if piece_type in piece_square_tables:
                    if piece.isupper():
                        # White piece => read table directly for (r, c)
                        psq_bonus = piece_square_tables[piece_type][r][c]
                        material_score += psq_bonus
                        if piece_type == 'B':
                            bishop_count_white += 1
                    else:
                        # Black piece => mirror row & col, then subtract
                        # so that "good" squares for Black reduce White’s evaluation
                        mirrored_bonus = piece_square_tables[piece_type][7 - r][7 - c]
                        material_score -= mirrored_bonus
                        if piece_type == 'B':
                            bishop_count_black += 1

        # bishop pair
        BISHOP_PAIR_BONUS = 0.5
        bishop_pair_score = 0
        if bishop_count_white >= 2:
            bishop_pair_score += BISHOP_PAIR_BONUS
        if bishop_count_black >= 2:
            bishop_pair_score -= BISHOP_PAIR_BONUS

        # Mobility
        white_moves = self.game.generate_all_moves('white', validate_check=False)
        black_moves = self.game.generate_all_moves('black', validate_check=False)
        mobility_score = (len(white_moves) - len(black_moves)) * 0.1

        # Evaluate other heuristics
        structure_score = self.evaluate_pawn_structure()
        rooks_score = self.evaluate_rooks_on_open_files()
        king_safety = self.evaluate_king_safety()

        # Advanced pawns
        ADVANCED_PAWN_BONUS = 0.1
        advanced_pawn_score = 0
        for row_idx in range(8):
            for col_idx in range(8):
                piece = self.game.board[row_idx][col_idx]
                if piece == 'P':
                    distance_from_start = 7 - row_idx
                    advanced_pawn_score += ADVANCED_PAWN_BONUS * distance_from_start
                elif piece == 'p':
                    distance_from_start = row_idx
                    advanced_pawn_score -= ADVANCED_PAWN_BONUS * distance_from_start

        # Undefended piece penalty
        undefended_penalty = self._penalize_undefended_pieces()

        # Sum up everything
        total_eval = (material_score
                    + bishop_pair_score
                    + mobility_score
                    + structure_score
                    + rooks_score
                    + king_safety
                    + advanced_pawn_score
                    + undefended_penalty)

        return total_eval

    def _get_piece_square_value(self, piece_type, row, col, is_white):
        """
        Returns a piece-square table bonus. 
        If is_white=False, we flip the row to account for black's perspective.
        """
        if is_white:
            return piece_square_tables[piece_type][row][col]
        else:
            # For black, we can flip row & col to mirror the table
            return piece_square_tables[piece_type][7 - row][col]

    def _penalize_undefended_pieces(self):
        """
        Adds a penalty if a piece can be captured on next move with no recapture.
        This is a simplified approach:
          - For each piece on the board, check if it's attacked by the opponent
            and not defended by any friendly piece.
        """
        penalty = 0.0
        # Opponent side
        opp_side = 'black' if self.game.turn == 'white' else 'white'
        # We'll generate all moves for both sides (without check validation, just to see attackers)
        opp_moves = self.game.generate_all_moves(opp_side, validate_check=False)
        my_moves = self.game.generate_all_moves(self.game.turn, validate_check=False)

        # Create sets of squares attacked by each side
        opp_attacks = set()
        my_attacks = set()
        for mv in opp_moves:
            # normal moves are (sr, sc, er, ec)
            if isinstance(mv, tuple) and len(mv) == 4:
                opp_attacks.add((mv[2], mv[3]))
        for mv in my_moves:
            if isinstance(mv, tuple) and len(mv) == 4:
                my_attacks.add((mv[2], mv[3]))

        # Now we check each piece on the board
        for r in range(8):
            for c in range(8):
                piece = self.game.board[r][c]
                if piece == '.' or piece.lower() == 'k':
                    continue  # ignore empty squares, or kings for now

                # If piece belongs to the side to move
                if (self.game.turn == 'white' and piece.isupper()) or \
                   (self.game.turn == 'black' and piece.islower()):
                    # It's "our" piece. If it is attacked by the opponent,
                    # but not defended by our side, apply a penalty.
                    if (r, c) in opp_attacks and (r, c) not in my_attacks:
                        # Increase penalty by piece value / 2, for example
                        penalty_value = abs(self.game.piece_values[piece]) * 0.5
                        penalty -= penalty_value

        return penalty

    def evaluate_pawn_structure(self):
        DOUBLED_PAWN_PENALTY = 0.2
        ISOLATED_PAWN_PENALTY = 0.25

        score = 0
        white_pawns_in_file = [0] * 8
        black_pawns_in_file = [0] * 8

        for row_idx in range(8):
            for col_idx in range(8):
                piece = self.game.board[row_idx][col_idx]
                if piece == 'P':
                    white_pawns_in_file[col_idx] += 1
                elif piece == 'p':
                    black_pawns_in_file[col_idx] += 1

        for col_idx in range(8):
            if white_pawns_in_file[col_idx] > 1:
                score -= DOUBLED_PAWN_PENALTY * (white_pawns_in_file[col_idx] - 1)
            if black_pawns_in_file[col_idx] > 1:
                score += DOUBLED_PAWN_PENALTY * (black_pawns_in_file[col_idx] - 1)

        for col_idx in range(8):
            if white_pawns_in_file[col_idx] > 0:
                left_file = col_idx - 1
                right_file = col_idx + 1
                no_left = (left_file < 0 or white_pawns_in_file[left_file] == 0)
                no_right = (right_file > 7 or white_pawns_in_file[right_file] == 0)
                if no_left and no_right:
                    score -= ISOLATED_PAWN_PENALTY * white_pawns_in_file[col_idx]

            if black_pawns_in_file[col_idx] > 0:
                left_file = col_idx - 1
                right_file = col_idx + 1
                no_left = (left_file < 0 or black_pawns_in_file[left_file] == 0)
                no_right = (right_file > 7 or black_pawns_in_file[right_file] == 0)
                if no_left and no_right:
                    score += ISOLATED_PAWN_PENALTY * black_pawns_in_file[col_idx]

        return score

    def evaluate_rooks_on_open_files(self):
        ROOK_OPEN_FILE_BONUS = 0.25
        ROOK_SEMI_OPEN_FILE_BONUS = 0.1

        score = 0
        white_pawns = [0] * 8
        black_pawns = [0] * 8
        for row in range(8):
            for col in range(8):
                if self.game.board[row][col] == 'P':
                    white_pawns[col] += 1
                elif self.game.board[row][col] == 'p':
                    black_pawns[col] += 1

        for row in range(8):
            for col in range(8):
                piece = self.game.board[row][col]
                if piece in ('R', 'r'):
                    file_has_white = (white_pawns[col] > 0)
                    file_has_black = (black_pawns[col] > 0)

                    if piece == 'R':
                        if not file_has_white and not file_has_black:
                            score += ROOK_OPEN_FILE_BONUS
                        elif not file_has_white and file_has_black:
                            score += ROOK_SEMI_OPEN_FILE_BONUS
                    else:  # 'r'
                        if not file_has_white and not file_has_black:
                            score -= ROOK_OPEN_FILE_BONUS
                        elif not file_has_black and file_has_white:
                            score -= ROOK_SEMI_OPEN_FILE_BONUS

        return score

    def evaluate_king_safety(self):
        KING_CASTLED_BONUS = 0.3
        KING_CENTER_PENALTY = 0.2
        KING_PAWN_COVER_BONUS = 0.1

        score = 0
        wk_row, wk_col = self.game.king_positions['white']
        if (wk_row, wk_col) in [(7, 6), (7, 2)]:
            score += KING_CASTLED_BONUS
        if 3 <= wk_row <= 4 and 3 <= wk_col <= 4:
            score -= KING_CENTER_PENALTY
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = wk_row + dr
                cc = wk_col + dc
                if self.game.is_within_bounds(rr, cc):
                    if self.game.board[rr][cc] == 'P':
                        score += KING_PAWN_COVER_BONUS

        bk_row, bk_col = self.game.king_positions['black']
        if (bk_row, bk_col) in [(0, 6), (0, 2)]:
            score -= KING_CASTLED_BONUS
        if 3 <= bk_row <= 4 and 3 <= bk_col <= 4:
            score += KING_CENTER_PENALTY
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = bk_row + dr
                cc = bk_col + dc
                if self.game.is_within_bounds(rr, cc):
                    if self.game.board[rr][cc] == 'p':
                        score -= KING_PAWN_COVER_BONUS

        return score

    def choose_best_promotion(self, sr, sc, er, ec, piece, board_before):
        if piece == 'P':
            promo_list = ['Q', 'R', 'B', 'N']
        else:
            promo_list = ['q', 'r', 'b', 'n']

        best_promo_eval = float('-inf') if self.side == 'white' else float('inf')
        best_promo_piece = promo_list[0]

        for promo in promo_list:
            self.game.board[sr][sc] = '.'
            self.game.board[er][ec] = promo
            promo_score = self.evaluate_position()
            self.game.board = copy.deepcopy(board_before)

            if self.side == 'white' and promo_score > best_promo_eval:
                best_promo_eval = promo_score
                best_promo_piece = promo
            elif self.side == 'black' and promo_score < best_promo_eval:
                best_promo_eval = promo_score
                best_promo_piece = promo

        return best_promo_piece, best_promo_eval

    def restore_game_state(self, board_copy_state,
                           turn_save,
                           king_positions_save,
                           castling_rights_save,
                           en_passant_save,
                           halfmove_clock_save):
        self.game.board = board_copy_state
        self.game.turn = turn_save
        self.game.king_positions = king_positions_save
        self.game.castling_rights = castling_rights_save
        self.game.en_passant_target = en_passant_save
        self.game.halfmove_clock = halfmove_clock_save
