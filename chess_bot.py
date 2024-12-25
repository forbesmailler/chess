import copy
import time
import random

class ChessBot:
    """
    A class responsible for choosing the best move for the bot side.
    It references an existing ChessGame object to read/update board state.
    """

    def __init__(self, game, side='black', max_depth=3, time_limit=30.0, random_range=0.05):
        self.game = game
        self.side = side
        self.max_depth = max_depth
        self.time_limit = time_limit
        
        # We'll add a random value in [-random_range, +random_range] to final eval
        # so the bot doesn't always pick the same move.
        self.random_range = random_range

        self.start_time = None
        self.debug = True

        # Simple transposition table: { (pos_key, depth): (flag, value, alpha, beta) }
        self.transposition_table = {}

    def choose_move(self):
        self.start_time = time.time()

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

        best_move, best_eval = self.alpha_beta_root(moves)
        if best_move:
            self._make_and_print_move(best_move)

    def alpha_beta_root(self, moves):
        """
        The root of alpha-beta: we do a shallow version of move-ordering here too.
        """
        # Move-order first
        moves = self.order_moves(moves, self.side)

        best_move = None
        if self.side == 'white':
            best_eval = float('-inf')
            for move in moves:
                if self._time_expired():
                    break
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=False)
                # random noise only at the root
                noise = random.uniform(-self.random_range, self.random_range)
                val_with_noise = val + noise

                if val_with_noise > best_eval:
                    best_eval = val_with_noise
                    best_move = move

                if self.debug:
                    print(f"[DEBUG] Move {move}, eval={val:.2f}, noisy={val_with_noise:.2f}")
        else:
            best_eval = float('inf')
            for move in moves:
                if self._time_expired():
                    break
                val = self.alpha_beta(move, depth=self.max_depth - 1,
                                      alpha=float('-inf'), beta=float('inf'),
                                      maximizing=True)
                # random noise only at the root
                noise = random.uniform(-self.random_range, self.random_range)
                val_with_noise = val + noise

                if val_with_noise < best_eval:
                    best_eval = val_with_noise
                    best_move = move

                if self.debug:
                    print(f"[DEBUG] Move {move}, eval={val:.2f}, noisy={val_with_noise:.2f}")

        return best_move, best_eval

    def alpha_beta(self, move, depth, alpha, beta, maximizing):
        """
        Alpha-beta with transposition table + move ordering.
        """
        # Save game state
        board_copy = copy.deepcopy(self.game.board)
        turn_save = self.game.turn
        king_positions_save = self.game.king_positions.copy()
        castling_rights_save = copy.deepcopy(self.game.castling_rights)
        en_passant_save = self.game.en_passant_target
        halfmove_clock_save = self.game.halfmove_clock

        # Make the move
        self.game.make_move(move)

        pos_key = (self.game.create_position_key(), depth)  # create_position_key must be stable

        # Check transposition table
        if pos_key in self.transposition_table:
            entry = self.transposition_table[pos_key]
            # e.g. could store an 'eval' and a 'flag' (exact, lowerbound, upperbound)
            # For simplicity, let's assume we stored an exact eval
            cached_eval = entry['value']
        else:
            cached_eval = None

        if cached_eval is not None:
            # We can skip searching deeper
            val = cached_eval
        else:
            # If time expired or depth=0 or game over => evaluate
            if (depth == 0 or self._time_expired() 
                or self.game.is_checkmate() or self.game.check_draw_conditions()):
                val = self.evaluate_position()
            else:
                # Next side
                next_side = self.game.turn
                next_moves = self.game.generate_all_moves(next_side, validate_check=True)
                if not next_moves:
                    val = self.evaluate_position()
                else:
                    # ORDER the next moves
                    next_moves = self.order_moves(next_moves, next_side)

                    if maximizing:
                        val = float('-inf')
                        for nxt_move in next_moves:
                            if self._time_expired():
                                break
                            current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, False)
                            val = max(val, current_eval)
                            alpha = max(alpha, val)
                            if beta <= alpha:
                                break
                    else:
                        val = float('inf')
                        for nxt_move in next_moves:
                            if self._time_expired():
                                break
                            current_eval = self.alpha_beta(nxt_move, depth-1, alpha, beta, True)
                            val = min(val, current_eval)
                            beta = min(beta, val)
                            if beta <= alpha:
                                break

            # Store in table
            self.transposition_table[pos_key] = {
                'value': val
                # you might store alpha/beta bounds, flags, etc. for advanced usage
            }

        # Restore game state
        self.restore_game_state(board_copy,
                                turn_save,
                                king_positions_save,
                                castling_rights_save,
                                en_passant_save,
                                halfmove_clock_save)

        return cached_eval if cached_eval is not None else val

    def order_moves(self, moves, side):
        """
        Basic move-ordering function:
        1) Captures first (maybe checks first if you detect them).
        2) Sort by piece value being captured, descending.
        3) Then non-captures afterwards.
        """
        scored_moves = []
        for m in moves:
            # A move might be a castling tuple or normal move
            if isinstance(m, tuple) and len(m) == 4:
                sr, sc, er, ec = m
                piece_captured = self.game.board[er][ec]
                if piece_captured != '.':
                    # It's a capture
                    captured_value = abs(self.game.piece_values.get(piece_captured, 0))
                    # higher is better for capturing big pieces
                    scored_moves.append((m, 100 + captured_value))
                else:
                    # Non-capture
                    scored_moves.append((m, 0))
            else:
                # If it's castling or something else, treat it as non-capture
                scored_moves.append((m, 0))

        # Sort in descending order if side is White (we want to see bigger values first),
        # ascending order if side is Black. 
        # Actually for alpha-beta, we typically do descending so we prune faster. 
        # We'll do descending in both cases for simplicity:
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        # Return the moves in new order
        ordered = [x[0] for x in scored_moves]
        return ordered

    def _time_expired(self):
        return (time.time() - self.start_time) >= self.time_limit

    # ---------------- UTILITY & EVALUATION ------------------------------------
    def _make_and_print_move(self, move):
        # Move is either ("castle", side, "kingside"/"queenside") or normal
        self.game.make_move(move)
        print(f"Bot plays: {self.game.convert_to_algebraic(move)}")

    def evaluate_position(self):
        # If checkmate => big +/- 
        if self.game.is_checkmate():
            return -9999 if self.game.turn == 'white' else 9999
        if self.game.check_draw_conditions():
            return 0

        material_score = 0
        bishop_count_white = 0
        bishop_count_black = 0
        for r in range(8):
            for c in range(8):
                piece = self.game.board[r][c]
                if piece != '.':
                    material_score += self.game.piece_values.get(piece, 0)

        # bishop pair
        BISHOP_PAIR_BONUS = 0.5
        bishop_pair_score = 0
        if bishop_count_white >= 2:
            bishop_pair_score += BISHOP_PAIR_BONUS
        if bishop_count_black >= 2:
            bishop_pair_score -= BISHOP_PAIR_BONUS

        # mobility
        white_moves = self.game.generate_all_moves('white', validate_check=False)
        black_moves = self.game.generate_all_moves('black', validate_check=False)
        mobility_score = (len(white_moves) - len(black_moves)) * 0.1

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

        # Sum up everything
        total_eval = (material_score
                    + bishop_pair_score
                    + mobility_score
                    + structure_score
                    + rooks_score
                    + king_safety
                    + advanced_pawn_score)

        return total_eval

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
