import copy
import time

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """

    def __init__(self, game, side='black', max_depth=3, time_limit=30.0):
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

        # We'll keep track of the start time each time we choose a move
        self.start_time = None

    def choose_move(self):
        """
        Finds the best move for self.side using a deeper alpha-beta
        search (up to self.max_depth). Also includes a time limit to avoid
        taking too long.
        """
        # Record when we started thinking about this move
        self.start_time = time.time()

        # Generate legal moves for this side
        moves = self.game.generate_all_moves(self.side, validate_check=True)
        if not moves:
            # No moves => checkmate or stalemate
            return

        # If we have time, run alpha-beta up to max_depth
        best_move = None
        if self.side == 'white':
            best_eval = float('-inf')
        else:
            best_eval = float('inf')

        # We do a quick check: if there's only one legal move, no need to search
        if len(moves) == 1:
            best_move = moves[0]
        else:
            # Search moves up to self.max_depth
            best_move, best_eval = self.alpha_beta_root(moves)

        # Execute the best move if found
        if best_move:
            # Handle castling
            if isinstance(best_move, tuple) and best_move[0] == "castle":
                self.game.make_move(best_move)
                print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")
                return

            # Handle promotions or normal moves
            sr, sc, er, ec = best_move
            piece = self.game.board[sr][sc]
            if (piece == 'P' and er == 0) or (piece == 'p' and er == 7):
                # We already pick the best promotion in alpha-beta, but if you still
                # want to do a secondary logic, you can handle promotions here.
                board_before = copy.deepcopy(self.game.board)
                best_promo_piece, _ = self.choose_best_promotion(
                    sr, sc, er, ec, piece, board_before
                )
                self.game.board = copy.deepcopy(board_before)
                self.game.board[sr][sc] = '.'
                self.game.board[er][ec] = best_promo_piece

                # Update halfmove clock, en passant, etc.
                if piece.lower() == 'p':
                    self.game.halfmove_clock = 0
                else:
                    self.game.halfmove_clock += 1
                self.game.en_passant_target = None
                self.game.record_position()
                print(f"Bot promotes to {best_promo_piece}!")

            self.game.make_move(best_move)
            print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")

    # -------------------------------------------------------------------------
    #                 ALPHA-BETA SEARCH ENTRY POINT
    # -------------------------------------------------------------------------
    def alpha_beta_root(self, moves):
        """
        Entry point for alpha-beta. We assume 'moves' is the list of legal moves
        for self.side. We'll iterate over them, do a search, and pick the best one.
        """
        best_move = None
        
        if self.side == 'white':
            best_eval = float('-inf')
            for move in moves:
                if self._time_expired():
                    break  # ran out of time
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=False)
                if val > best_eval:
                    best_eval = val
                    best_move = move
        else:
            best_eval = float('inf')
            for move in moves:
                if self._time_expired():
                    break  # ran out of time
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=True)
                if val < best_eval:
                    best_eval = val
                    best_move = move

        return best_move, best_eval

    def alpha_beta(self, move, depth, alpha, beta, maximizing):
        """
        Perform one step of alpha-beta for the given 'move'. That is:
        1) Make 'move'.
        2) Recursively search up to (depth) more plies.
        3) Revert 'move'.
        4) Return the best evaluation from that subtree.
        """
        # Save state
        board_copy_state = copy.deepcopy(self.game.board)
        turn_save = self.game.turn
        king_positions_save = self.game.king_positions.copy()
        castling_rights_save = copy.deepcopy(self.game.castling_rights)
        en_passant_save = self.game.en_passant_target
        halfmove_clock_save = self.game.halfmove_clock

        # Make the move
        self.game.make_move(move)

        # If time expired or depth=0, or game over => evaluate and revert
        if depth == 0 or self._time_expired() or self.game.is_checkmate() or self.game.check_draw_conditions():
            val = self.evaluate_position()
            self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                    castling_rights_save, en_passant_save, halfmove_clock_save)
            return val

        # Now generate moves for the next side
        next_side = 'white' if self.game.turn == 'white' else 'black'
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
                    break  # beta cut-off
            self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                    castling_rights_save, en_passant_save, halfmove_clock_save)
            return best_eval
        else:
            best_eval = float('inf')
            for nxt_move in moves:
                if self._time_expired():
                    break
                current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, True)
                best_eval = min(best_eval, current_eval)
                beta = min(beta, best_eval)
                if beta <= alpha:
                    break  # alpha cut-off
            self.restore_game_state(board_copy_state, turn_save, king_positions_save,
                                    castling_rights_save, en_passant_save, halfmove_clock_save)
            return best_eval

    def _time_expired(self):
        """Return True if we've exceeded our self.time_limit (in seconds)."""
        if (time.time() - self.start_time) >= self.time_limit:
            return True
        return False

    # -------------------------------------------------------------------------
    #          EVERYTHING ELSE (evaluate_position, promotions, etc.)
    # -------------------------------------------------------------------------
    def evaluate_position(self):
        """
        Existing evaluation function from your code (slightly truncated for brevity).
        """
        # 1. Checkmate / Draw
        if self.game.is_checkmate():
            return -9999 if self.game.turn == 'white' else 9999
        if self.game.check_draw_conditions():
            return 0

        # 2. Material + bishop pair
        material_score = 0
        bishop_count_white = 0
        bishop_count_black = 0
        for row in self.game.board:
            for piece in row:
                material_score += self.game.piece_values.get(piece, 0)
                if piece == 'B':
                    bishop_count_white += 1
                elif piece == 'b':
                    bishop_count_black += 1
        # bishop pair
        bishop_pair_score = 0
        BISHOP_PAIR_BONUS = 0.5
        if bishop_count_white >= 2:
            bishop_pair_score += BISHOP_PAIR_BONUS
        if bishop_count_black >= 2:
            bishop_pair_score -= BISHOP_PAIR_BONUS

        # 3. Mobility
        white_moves = self.game.generate_all_moves('white', validate_check=False)
        black_moves = self.game.generate_all_moves('black', validate_check=False)
        mobility_score = (len(white_moves) - len(black_moves)) * 0.1

        # 4. Add your other evaluation terms (advanced pawns, rook files, king safety, etc.)
        structure_score = self.evaluate_pawn_structure()
        rooks_score = self.evaluate_rooks_on_open_files()
        king_safety = self.evaluate_king_safety()

        # 5. Advanced Pawns bonus
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

        return (material_score
                + bishop_pair_score
                + mobility_score
                + structure_score
                + rooks_score
                + king_safety
                + advanced_pawn_score)

    def evaluate_pawn_structure(self):
        """
        Simplified pawn-structure evaluation (doubled/isolated).
        A positive return value favors White, negative favors Black.
        """
        DOUBLED_PAWN_PENALTY = 0.2
        ISOLATED_PAWN_PENALTY = 0.25

        score = 0

        # Identify pawns by files
        white_pawns_in_file = [0] * 8
        black_pawns_in_file = [0] * 8

        for row_idx in range(8):
            for col_idx in range(8):
                piece = self.game.board[row_idx][col_idx]
                if piece == 'P':
                    white_pawns_in_file[col_idx] += 1
                elif piece == 'p':
                    black_pawns_in_file[col_idx] += 1

        # Calculate doubled pawns
        for col_idx in range(8):
            if white_pawns_in_file[col_idx] > 1:
                score -= DOUBLED_PAWN_PENALTY * (white_pawns_in_file[col_idx] - 1)
            if black_pawns_in_file[col_idx] > 1:
                score += DOUBLED_PAWN_PENALTY * (black_pawns_in_file[col_idx] - 1)

        # Check for isolated pawns (no pawns in adjacent files)
        for col_idx in range(8):
            # White
            if white_pawns_in_file[col_idx] > 0:
                left_file = col_idx - 1
                right_file = col_idx + 1
                no_white_left = (left_file < 0 or white_pawns_in_file[left_file] == 0)
                no_white_right = (right_file > 7 or white_pawns_in_file[right_file] == 0)
                if no_white_left and no_white_right:
                    score -= ISOLATED_PAWN_PENALTY * white_pawns_in_file[col_idx]

            # Black
            if black_pawns_in_file[col_idx] > 0:
                left_file = col_idx - 1
                right_file = col_idx + 1
                no_black_left = (left_file < 0 or black_pawns_in_file[left_file] == 0)
                no_black_right = (right_file > 7 or black_pawns_in_file[right_file] == 0)
                if no_black_left and no_black_right:
                    score += ISOLATED_PAWN_PENALTY * black_pawns_in_file[col_idx]

        return score

    def evaluate_rooks_on_open_files(self):
        """
        Give a bonus to rooks on open or semi-open files.
        """
        ROOK_OPEN_FILE_BONUS = 0.25
        ROOK_SEMI_OPEN_FILE_BONUS = 0.1

        score = 0

        # Collect all pawns by file
        white_pawns = [0] * 8
        black_pawns = [0] * 8
        for row in range(8):
            for col in range(8):
                if self.game.board[row][col] == 'P':
                    white_pawns[col] += 1
                elif self.game.board[row][col] == 'p':
                    black_pawns[col] += 1

        # Check rooks
        for row in range(8):
            for col in range(8):
                piece = self.game.board[row][col]
                if piece in ('R', 'r'):
                    file_has_white_pawns = (white_pawns[col] > 0)
                    file_has_black_pawns = (black_pawns[col] > 0)

                    if piece == 'R':
                        # White rook
                        if not file_has_white_pawns and not file_has_black_pawns:
                            score += ROOK_OPEN_FILE_BONUS
                        elif not file_has_white_pawns and file_has_black_pawns:
                            score += ROOK_SEMI_OPEN_FILE_BONUS
                    else:
                        # Black rook
                        if not file_has_white_pawns and not file_has_black_pawns:
                            score -= ROOK_OPEN_FILE_BONUS
                        elif not file_has_black_pawns and file_has_white_pawns:
                            score -= ROOK_SEMI_OPEN_FILE_BONUS

        return score

    def evaluate_king_safety(self):
        """
        Give a simple measure of king safety.
        """
        KING_CASTLED_BONUS = 0.3
        KING_CENTER_PENALTY = 0.2
        KING_PAWN_COVER_BONUS = 0.1

        score = 0

        # White king
        wk_row, wk_col = self.game.king_positions['white']
        # If white castled, the king should be on g1 or c1
        if (wk_row, wk_col) in [(7, 6), (7, 2)]:
            score += KING_CASTLED_BONUS

        # Center penalty
        if 3 <= wk_row <= 4 and 3 <= wk_col <= 4:
            score -= KING_CENTER_PENALTY

        # Pawn cover
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = wk_row + dr
                cc = wk_col + dc
                if self.game.is_within_bounds(rr, cc):
                    if self.game.board[rr][cc] == 'P':
                        score += KING_PAWN_COVER_BONUS

        # Black king
        bk_row, bk_col = self.game.king_positions['black']
        if (bk_row, bk_col) in [(0, 6), (0, 2)]:
            score -= KING_CASTLED_BONUS  # negative from White perspective

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

    def simulate_two_ply(self, move):
        """
        Apply 'move' for bot_side, then let the opponent pick its best move.
        Return the final position evaluation (from White's perspective).
        """
        # 1. Save state for revert
        board_copy_state = copy.deepcopy(self.game.board)
        turn_save = self.game.turn
        king_positions_save = self.game.king_positions.copy()
        castling_rights_save = copy.deepcopy(self.game.castling_rights)
        en_passant_save = self.game.en_passant_target
        halfmove_clock_save = self.game.halfmove_clock

        # 2. Make the bot's move
        self.game.make_move(move)

        opponent_side = 'black' if self.side == 'white' else 'white'
        # If no opponent moves => either checkmate or stalemate, so just evaluate
        opponent_moves = self.game.generate_all_moves(opponent_side, validate_check=True)
        if not opponent_moves:
            final_eval = self.evaluate_position()
            # revert
            self.restore_game_state(board_copy_state,
                                    turn_save,
                                    king_positions_save,
                                    castling_rights_save,
                                    en_passant_save,
                                    halfmove_clock_save)
            return final_eval

        # 3. Opponent picks its best move from its perspective
        best_eval_for_opponent = float('-inf') if opponent_side == 'white' else float('inf')

        for op_move in opponent_moves:
            temp_board = copy.deepcopy(self.game.board)
            temp_turn = self.game.turn
            temp_king_pos = self.game.king_positions.copy()
            temp_castling = copy.deepcopy(self.game.castling_rights)
            temp_en_passant = self.game.en_passant_target
            temp_halfmove = self.game.halfmove_clock

            self.game.make_move(op_move)
            current_eval = self.evaluate_position()

            # Revert
            self.game.board = temp_board
            self.game.turn = temp_turn
            self.game.king_positions = temp_king_pos
            self.game.castling_rights = temp_castling
            self.game.en_passant_target = temp_en_passant
            self.game.halfmove_clock = temp_halfmove

            # Opponent tries to produce a favorable eval for themselves
            if opponent_side == 'white':
                # Opponent is White => maximize
                if current_eval > best_eval_for_opponent:
                    best_eval_for_opponent = current_eval
            else:
                # Opponent is Black => minimize
                if current_eval < best_eval_for_opponent:
                    best_eval_for_opponent = current_eval

        # 4. Simulate applying that best_opponent_move "for real"
        #    Actually we don't know the exact best_opponent_move here without storing it.
        #    But to get the final_eval, let's just assume that's the position after
        #    best_eval_for_opponent is achieved. We can just do a single final eval
        #    after a typical opponent move. However, a thorough approach would track
        #    the best_opponent_move as well.
        #
        # For simplicity, we just evaluate the position *after* the first move
        # we made, ignoring the exact move the opponent picks, because we already
        # took the min/max result.
        final_eval = best_eval_for_opponent

        # Revert everything
        self.restore_game_state(board_copy_state,
                                turn_save,
                                king_positions_save,
                                castling_rights_save,
                                en_passant_save,
                                halfmove_clock_save)

        return final_eval

    def choose_best_promotion(self, start_row, start_col, end_row, end_col,
                              piece, board_before):
        if piece == 'P':
            promo_list = ['Q', 'R', 'B', 'N']
        else:
            promo_list = ['q', 'r', 'b', 'n']

        best_promo_eval = float('-inf') if self.side == 'white' else float('inf')
        best_promo_piece = promo_list[0]

        for promo in promo_list:
            self.game.board[start_row][start_col] = '.'
            self.game.board[end_row][end_col] = promo
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
        """Helper to revert the game state after a temporary simulation."""
        self.game.board = board_copy_state
        self.game.turn = turn_save
        self.game.king_positions = king_positions_save
        self.game.castling_rights = castling_rights_save
        self.game.en_passant_target = en_passant_save
        self.game.halfmove_clock = halfmove_clock_save
