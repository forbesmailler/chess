import copy

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """

    def __init__(self, game, side='black'):
        self.game = game  # reference to an instance of ChessGame
        self.side = side  # 'white' or 'black'

    def choose_move(self):
        """
        Finds the best move for self.side using a simple 2-ply lookahead
        (simulate_two_ply). Then executes it on the game board.
        """
        moves = self.game.generate_all_moves(self.side, validate_check=True)
        if not moves:
            # No moves => checkmate or stalemate
            return

        best_move = None
        # White wants to maximize the evaluation; Black wants to minimize
        best_score = float('-inf') if self.side == 'white' else float('inf')

        for move in moves:
            # Some moves may be castling or normal piece moves
            if isinstance(move, tuple) and move[0] == "castle":
                # Evaluate castling quickly by simulating
                final_score = self.simulate_two_ply(move)
            else:
                start_row, start_col, end_row, end_col = move
                piece = self.game.board[start_row][start_col]

                # If it's a potential promotion for the bot, test all promotion candidates
                if (piece == 'P' and end_row == 0) or (piece == 'p' and end_row == 7):
                    final_score = self.handle_bot_promotions(
                        start_row, start_col, end_row, end_col, piece
                    )
                else:
                    final_score = self.simulate_two_ply(move)

            # Track best or worst move depending on side
            if self.side == 'white' and final_score > best_score:
                best_score = final_score
                best_move = move
            elif self.side == 'black' and final_score < best_score:
                best_score = final_score
                best_move = move

        # Execute the best move if found
        if best_move:
            if isinstance(best_move, tuple) and best_move[0] == "castle":
                self.game.make_move(best_move)
                print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")
            else:
                sr, sc, er, ec = best_move
                piece = self.game.board[sr][sc]
                # Double-check promotion
                if (piece == 'P' and er == 0) or (piece == 'p' and er == 7):
                    # We already picked best promotion in handle_bot_promotions,
                    # so just finalize it with the best piece found.
                    board_before = copy.deepcopy(self.game.board)
                    best_promo_piece, _ = self.choose_best_promotion(
                        sr, sc, er, ec, piece, board_before
                    )
                    self.game.board = copy.deepcopy(board_before)
                    self.game.board[sr][sc] = '.'
                    self.game.board[er][ec] = best_promo_piece

                    # En passant, halfmove clock, etc.:
                    if piece.lower() == 'p':
                        self.game.halfmove_clock = 0
                    else:
                        self.game.halfmove_clock += 1
                    self.game.en_passant_target = None
                    self.game.record_position()

                    print(f"Bot promotes to {best_promo_piece}!")
                self.game.make_move(best_move)
                print(f"Bot plays: {self.game.convert_to_algebraic(best_move)}")

    def evaluate_position(self):
        """
        ADVANCED EVALUATION FUNCTION (placeholder).
        Returns a numeric evaluation from White's perspective.
        """
        # 1. Checkmate/Draw checks
        if self.game.is_checkmate():
            # The side to move is checkmated => big negative from their perspective
            return -9999 if self.game.turn == 'white' else 9999
        if self.game.check_draw_conditions():
            return 0  # Draw

        # 2. Material evaluation
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

        # 3. Mobility
        white_moves = self.game.generate_all_moves('white', validate_check=False)
        black_moves = self.game.generate_all_moves('black', validate_check=False)
        mobility_score = (len(white_moves) - len(black_moves)) * 0.1

        # 4. Bishop pair bonus
        BISHOP_PAIR_BONUS = 0.5
        bishop_pair_score = 0
        if bishop_count_white >= 2:
            bishop_pair_score += BISHOP_PAIR_BONUS
        if bishop_count_black >= 2:
            bishop_pair_score -= BISHOP_PAIR_BONUS

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

        # Combine
        total_evaluation = (
            material_score
            + mobility_score
            + bishop_pair_score
            + self.evaluate_pawn_structure()
            + self.evaluate_rooks_on_open_files()
            + self.evaluate_king_safety()
        )
        return total_evaluation

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
    
    def handle_bot_promotions(self, sr, sc, er, ec, piece):
        """
        If a pawn is about to promote, try all promotion pieces and pick the one
        giving the best 2-ply result.
        """
        board_copy_state = copy.deepcopy(self.game.board)
        # Temporarily evaluate all promotions
        best_score = float('-inf') if self.side == 'white' else float('inf')

        promo_candidates = ['Q', 'R', 'B', 'N'] if piece == 'P' else ['q', 'r', 'b', 'n']
        for promo in promo_candidates:
            # Make the promotion
            self.game.board[sr][sc] = '.'
            self.game.board[er][ec] = promo

            # Evaluate 2-ply
            candidate_score = self.simulate_two_ply((sr, sc, er, ec))

            # Revert
            self.game.board = copy.deepcopy(board_copy_state)

            if self.side == 'white' and candidate_score > best_score:
                best_score = candidate_score
            elif self.side == 'black' and candidate_score < best_score:
                best_score = candidate_score

        # Restore the board
        self.game.board = copy.deepcopy(board_copy_state)
        return best_score

    def choose_best_promotion(self, start_row, start_col, end_row, end_col,
                              piece, board_before):
        """
        Temporarily tries all possible promotion pieces and returns:
        (best_promo_piece, best_promo_eval).
        This is used by choose_move() to finalize the actual promotion piece.
        """
        if piece == 'P':
            promo_list = ['Q', 'R', 'B', 'N']
        else:
            promo_list = ['q', 'r', 'b', 'n']

        best_promo_eval = float('-inf') if self.side == 'white' else float('inf')
        best_promo_piece = promo_list[0]

        for promo in promo_list:
            # Make the promotion on a temp board
            self.game.board[start_row][start_col] = '.'
            self.game.board[end_row][end_col] = promo

            promo_score = self.evaluate_position()

            # Restore board
            self.game.board = copy.deepcopy(board_before)

            # Track best/worst eval depending on side
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
