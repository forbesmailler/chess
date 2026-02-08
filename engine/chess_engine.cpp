#include "chess_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

ChessEngine::ChessEngine(int max_time_ms, EvalMode eval_mode,
                         std::shared_ptr<NNUEModel> nnue_model)
    : BaseEngine(max_time_ms, eval_mode, std::move(nnue_model)) {
    eval_cache.resize(EVAL_CACHE_SIZE);
    transposition_table.resize(TT_SIZE);
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(countermoves, 0, sizeof(countermoves));
}

float ChessEngine::evaluate(const ChessBoard& board) {
    uint64_t pos_key = board.hash();
    auto& entry = eval_cache[pos_key & EVAL_CACHE_MASK];
    if (entry.key == pos_key) return entry.score;

    float eval = raw_evaluate(board);
    entry = {pos_key, eval};
    return eval;
}

static void pick_move(chess::Movelist& moves, int* scores, int start, int count) {
    int best_idx = start;
    for (int i = start + 1; i < count; ++i) {
        if (scores[i] > scores[best_idx]) best_idx = i;
    }
    if (best_idx != start) {
        std::swap(moves[start], moves[best_idx]);
        std::swap(scores[start], scores[best_idx]);
    }
}

void ChessEngine::order_moves(const ChessBoard& board, chess::Movelist& moves,
                              chess::Move tt_move, int ply, chess::Move prev_move) {
    if (moves.size() <= 1) return;

    int scores[256];
    int n = moves.size();
    score_moves(board, moves, scores, tt_move, ply, prev_move);

    int idx[256];
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&scores](int a, int b) { return scores[a] > scores[b]; });

    chess::Move temp[256];
    for (int i = 0; i < n; ++i) temp[i] = moves[idx[i]];
    for (int i = 0; i < n; ++i) moves[i] = temp[i];
}

void ChessEngine::score_moves(const ChessBoard& board, const chess::Movelist& moves,
                              int* scores, chess::Move tt_move, int ply,
                              chess::Move prev_move) {
    constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900, 0, 0};
    bool has_tt_move = tt_move != chess::Move::NO_MOVE;
    bool has_prev = prev_move != chess::Move::NO_MOVE;
    chess::Move cm = chess::Move::NO_MOVE;
    if (has_prev) {
        cm = countermoves[prev_move.from().index()][prev_move.to().index()];
    }

    for (int i = 0; i < moves.size(); ++i) {
        chess::Move move = moves[i];
        int score = 0;
        bool is_capture = move.typeOf() == chess::Move::ENPASSANT ||
                          board.board.at(move.to()) != chess::Piece::NONE;

        if (has_tt_move && move == tt_move) {
            score = 1000000;
        } else if (is_capture) {
            int victim =
                PIECE_VALUES[static_cast<int>(board.board.at(move.to()).type())];
            int attacker =
                PIECE_VALUES[static_cast<int>(board.board.at(move.from()).type())];
            score = 100000 + victim * 10 - attacker;
        } else if (move.typeOf() == chess::Move::PROMOTION) {
            score = 90000;
        } else {
            if (ply < MAX_PLY) {
                if (move == killers[ply][0]) {
                    score = 80000;
                } else if (move == killers[ply][1]) {
                    score = 70000;
                }
            }
            if (score == 0 && cm != chess::Move::NO_MOVE && move == cm) {
                score = 60000;
            }
            if (score == 0) {
                score = history[move.from().index()][move.to().index()];
            }
        }
        scores[i] = score;
    }
}

SearchResult ChessEngine::get_best_move(const ChessBoard& board,
                                        const TimeControl& time_control) {
    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board.board);

    if (legal_moves.empty()) {
        float score = board.board.inCheck() ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }
    if (legal_moves.size() == 1) {
        ChessBoard::Move m;
        m.internal_move = legal_moves[0];
        return {m, 0.0f, 1, std::chrono::milliseconds(50), 1};
    }

    return iterative_deepening_search(board, calculate_search_time(time_control));
}

float ChessEngine::negamax(ChessBoard& board, int depth, int ply, float alpha,
                           float beta, bool is_pv, chess::Move prev_move) {
    if ((nodes_searched.fetch_add(1) & (TIME_CHECK_INTERVAL - 1)) == 0) check_time();
    if (should_stop.load()) return SEARCH_INTERRUPTED;

    {
        auto [go_reason, go_result] = board.board.isGameOver();
        if (go_result != chess::GameResult::NONE) {
            return go_reason == chess::GameResultReason::CHECKMATE ? -MATE_VALUE : 0.0f;
        }
    }

    uint64_t pos_key = board.hash();
    chess::Move tt_move = chess::Move::NO_MOVE;
    const auto& tt_entry = transposition_table[pos_key & TT_MASK];

    if (tt_entry.key == pos_key) {
        tt_move = tt_entry.best_move;

        if (tt_entry.depth >= depth && !is_pv) {
            switch (tt_entry.type) {
                case TranspositionEntry::EXACT:
                    return tt_entry.score;
                case TranspositionEntry::LOWER_BOUND:
                    if (tt_entry.score >= beta) return tt_entry.score;
                    alpha = std::max(alpha, tt_entry.score);
                    break;
                case TranspositionEntry::UPPER_BOUND:
                    if (tt_entry.score <= alpha) return tt_entry.score;
                    beta = std::min(beta, tt_entry.score);
                    break;
            }
            if (alpha >= beta) return tt_entry.score;
        }
    }

    bool in_check = board.board.inCheck();

    // Check extension — search deeper when in check
    if (in_check && depth == 0) depth = 1;

    if (depth == 0) {
        return quiescence_search(board, alpha, beta, 0, in_check);
    }

    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board.board);
    if (legal_moves.empty()) {
        return in_check ? -MATE_VALUE : 0.0f;
    }

    // Static eval for pruning decisions (reverse futility + futility)
    float static_eval = 0.0f;
    bool have_static_eval = !in_check && !is_pv;
    if (have_static_eval) {
        static_eval = evaluate(board);
        static_eval = board.turn() == ChessBoard::WHITE ? static_eval : -static_eval;

        // Reverse futility pruning (static null move pruning)
        if (depth <= 3 && static_eval - depth * 1500.0f >= beta) {
            return static_eval;
        }
    }

    // Null Move Pruning — make/unmake in place (no board copy)
    if (depth > config::search::null_move::MIN_DEPTH && !is_pv && !in_check &&
        beta < MATE_VALUE - config::search::null_move::MATE_MARGIN &&
        alpha > -MATE_VALUE + config::search::null_move::MATE_MARGIN) {
        int reduction = depth > config::search::null_move::DEEP_THRESHOLD
                            ? config::search::null_move::DEEP_REDUCTION
                            : config::search::null_move::SHALLOW_REDUCTION;
        int null_depth = depth - 1 - reduction;

        if (null_depth >= 0) {
            board.board.makeNullMove();
            float null_score =
                -negamax(board, null_depth, ply + 1, -beta, -beta + 1, false);
            board.board.unmakeNullMove();
            if (null_score >= beta) {
                return null_score;
            }
        }
    }

    // Score moves for incremental selection (avoid full sort)
    int scores[256];
    score_moves(board, legal_moves, scores, tt_move, ply, prev_move);
    float best_score = -std::numeric_limits<float>::infinity();
    chess::Move best_move = chess::Move::NO_MOVE;
    auto node_type = TranspositionEntry::UPPER_BOUND;

    int moves_searched = 0;
    int num_moves = legal_moves.size();
    for (int mi = 0; mi < num_moves; ++mi) {
        pick_move(legal_moves, scores, mi, num_moves);
        chess::Move move = legal_moves[mi];
        bool is_capture = move.typeOf() == chess::Move::ENPASSANT ||
                          board.board.at(move.to()) != chess::Piece::NONE;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;
        bool is_quiet = !is_capture && !is_promo;

        // Futility pruning: skip quiet moves unlikely to raise alpha
        if (have_static_eval && depth <= 2 && is_quiet && moves_searched > 0) {
            float margin = depth == 1 ? 1500.0f : 3000.0f;
            if (static_eval + margin <= alpha) continue;
        }

        // Late move pruning: skip quiet late moves at shallow depth
        if (!in_check && !is_pv && depth <= 3 && is_quiet && moves_searched > 0) {
            if (moves_searched >= 3 + depth * depth) continue;
        }

        board.board.makeMove(move);
        moves_searched++;
        float score;

        if (moves_searched == 1) {
            // First move (PV): full window, full depth
            score = -negamax(board, depth - 1, ply + 1, -beta, -alpha, is_pv, move);
        } else {
            // LMR for late quiet moves
            bool do_lmr = moves_searched > config::search::lmr::MIN_MOVES_SEARCHED &&
                          depth > config::search::lmr::MIN_DEPTH && !in_check &&
                          is_quiet;

            if (do_lmr) {
                int reduction =
                    moves_searched > config::search::lmr::MANY_MOVES_THRESHOLD
                        ? config::search::lmr::DEEP_REDUCTION
                        : config::search::lmr::SHALLOW_REDUCTION;
                int reduced_depth = depth - 1 - reduction;

                if (reduced_depth >= 0) {
                    score = -negamax(board, reduced_depth, ply + 1, -alpha - 1, -alpha,
                                     false, move);
                } else {
                    score = -negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha,
                                     false, move);
                }
            } else {
                score = -negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha, false,
                                 move);
            }

            // PVS re-search: if null window failed high, re-search with full window
            if (score > alpha && score < beta) {
                score = -negamax(board, depth - 1, ply + 1, -beta, -alpha, true, move);
            }
        }

        board.board.unmakeMove(move);

        if (score == SEARCH_INTERRUPTED) {
            return SEARCH_INTERRUPTED;
        }

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }

        if (score >= beta) {
            node_type = TranspositionEntry::LOWER_BOUND;

            // Update killer moves, history, and countermoves for quiet cutoffs
            if (is_quiet && ply < MAX_PLY) {
                if (move != killers[ply][0]) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = move;
                }
                history[move.from().index()][move.to().index()] += depth * depth;
                if (prev_move != chess::Move::NO_MOVE) {
                    countermoves[prev_move.from().index()][prev_move.to().index()] =
                        move;
                }
            }
            break;
        }

        if (score > alpha) {
            alpha = score;
            node_type = TranspositionEntry::EXACT;
        }
    }

    {
        auto& tt_slot = transposition_table[pos_key & TT_MASK];
        int new_value = depth * 4 + (node_type == TranspositionEntry::EXACT ? 2 : 0);
        int old_value =
            tt_slot.depth * 4 + (tt_slot.type == TranspositionEntry::EXACT ? 2 : 0);
        if (tt_slot.key == pos_key || new_value >= old_value) {
            tt_slot = {pos_key, best_score, depth, node_type, best_move};
        }
    }

    return best_score;
}

float ChessEngine::quiescence_search(ChessBoard& board, float alpha, float beta,
                                     int qs_depth, bool in_check) {
    if (should_stop.load()) return SEARCH_INTERRUPTED;

    float stand_pat = evaluate(board);
    auto stm = board.turn();
    stand_pat = stm == ChessBoard::WHITE ? stand_pat : -stand_pat;

    if (qs_depth >= config::search::QUIESCENCE_MAX_DEPTH) return stand_pat;

    if (!in_check) {
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    chess::Movelist tactical_moves;
    if (in_check) {
        chess::movegen::legalmoves(tactical_moves, board.board);
        if (tactical_moves.empty()) return -MATE_VALUE;
    } else {
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(tactical_moves,
                                                                         board.board);
    }

    if (!tactical_moves.empty()) {
        constexpr float QS_PIECE_VALUES[] = {1000.0f, 3200.0f, 3300.0f, 5000.0f,
                                             9000.0f, 0.0f,    0.0f};
        constexpr float QS_DELTA_MARGIN = 2000.0f;

        order_moves(board, tactical_moves, chess::Move::NO_MOVE, 0);

        for (int i = 0; i < tactical_moves.size(); ++i) {
            chess::Move move = tactical_moves[i];

            // Delta pruning: skip captures that can't raise alpha
            if (!in_check && move.typeOf() != chess::Move::PROMOTION) {
                int pt = static_cast<int>(board.board.at(move.to()).type());
                float victim = QS_PIECE_VALUES[pt];
                if (victim == 0.0f) victim = 1000.0f;  // en passant
                if (stand_pat + victim + QS_DELTA_MARGIN < alpha) continue;
            }

            board.board.makeMove(move);
            bool child_in_check = board.board.inCheck();
            float score =
                -quiescence_search(board, -beta, -alpha, qs_depth + 1, child_in_check);
            board.board.unmakeMove(move);

            if (score == SEARCH_INTERRUPTED) return SEARCH_INTERRUPTED;
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
    }

    return alpha;
}

void ChessEngine::check_time() {
    if (std::chrono::steady_clock::now() >= search_deadline) {
        should_stop.store(true);
    }
}

SearchResult ChessEngine::iterative_deepening_search(ChessBoard board,
                                                     int max_time_ms) {
    auto start_time = std::chrono::steady_clock::now();
    search_deadline = start_time + std::chrono::milliseconds(max_time_ms);
    should_stop.store(false);
    nodes_searched.store(0);

    chess::Movelist legal_moves;
    chess::movegen::legalmoves(legal_moves, board.board);
    auto stm = board.turn();
    if (legal_moves.empty()) {
        float score = board.board.inCheck() ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }

    float static_eval = evaluate(board);
    static_eval = stm == ChessBoard::WHITE ? static_eval : -static_eval;
    ChessBoard::Move first_move;
    first_move.internal_move = legal_moves[0];
    SearchResult best_result{first_move, static_eval, 0, {}, 0};
    uint64_t pos_key = board.hash();
    float prev_score = static_eval;

    for (int depth = 1; depth <= config::search::MAX_DEPTH; ++depth) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= max_time_ms) break;

        chess::Move tt_move = chess::Move::NO_MOVE;
        {
            const auto& tt_entry = transposition_table[pos_key & TT_MASK];
            if (tt_entry.key == pos_key) tt_move = tt_entry.best_move;
        }

        order_moves(board, legal_moves, tt_move, 0);

        // Aspiration windows: use narrow window around previous score for depth >= 3
        float alpha, beta;
        if (depth >= 3) {
            alpha = prev_score - ASPIRATION_DELTA;
            beta = prev_score + ASPIRATION_DELTA;
        } else {
            alpha = -std::numeric_limits<float>::infinity();
            beta = std::numeric_limits<float>::infinity();
        }

        bool completed_depth = true;

        for (int attempt = 0;; ++attempt) {
            chess::Move current_best_move = legal_moves[0];
            float current_best_score = -std::numeric_limits<float>::infinity();
            int move_count = 0;

            for (int mi = 0; mi < legal_moves.size(); ++mi) {
                if (should_stop.load()) {
                    completed_depth = false;
                    break;
                }

                chess::Move move = legal_moves[mi];
                board.board.makeMove(move);
                move_count++;

                float score;
                if (move_count == 1) {
                    score = -negamax(board, depth - 1, 1, -beta, -alpha, true, move);
                } else {
                    score =
                        -negamax(board, depth - 1, 1, -alpha - 1, -alpha, false, move);
                    if (score > alpha && score < beta) {
                        score =
                            -negamax(board, depth - 1, 1, -beta, -alpha, true, move);
                    }
                }
                board.board.unmakeMove(move);

                if (score == -SEARCH_INTERRUPTED || should_stop.load()) {
                    completed_depth = false;
                    break;
                }

                if (score > current_best_score) {
                    current_best_score = score;
                    current_best_move = move;
                }

                alpha = std::max(alpha, score);
                if (alpha >= beta) break;
            }

            if (!completed_depth || should_stop.load()) break;

            // If aspiration window failed, widen and retry once
            if (attempt == 0 && depth >= 3 &&
                (current_best_score <= prev_score - ASPIRATION_DELTA ||
                 current_best_score >= prev_score + ASPIRATION_DELTA)) {
                alpha = -std::numeric_limits<float>::infinity();
                beta = std::numeric_limits<float>::infinity();
                order_moves(board, legal_moves, tt_move, 0);
                continue;
            }

            // Successful completion — save results
            ChessBoard::Move best_move_wrapper;
            best_move_wrapper.internal_move = current_best_move;
            best_result = {best_move_wrapper,
                           current_best_score,
                           depth,
                           {},
                           nodes_searched.load()};
            prev_score = current_best_score;

            transposition_table[pos_key & TT_MASK] = {pos_key, current_best_score,
                                                      depth, TranspositionEntry::EXACT,
                                                      current_best_move};
            break;
        }

        if (!completed_depth) break;
        if (std::abs(prev_score) > MATE_VALUE - 500) break;
    }

    best_result.time_used = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);

    return best_result;
}

void ChessEngine::clear_caches() {
    std::memset(eval_cache.data(), 0, EVAL_CACHE_SIZE * sizeof(EvalCacheEntry));
    transposition_table.assign(TT_SIZE, TranspositionEntry{});
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(countermoves, 0, sizeof(countermoves));
}
