#include "chess_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

ChessEngine::ChessEngine(int max_time_ms, EvalMode eval_mode,
                         std::shared_ptr<NNUEModel> nnue_model)
    : BaseEngine(max_time_ms, eval_mode, std::move(nnue_model)) {
    eval_cache.reserve(CACHE_SIZE / 2);
    transposition_table.reserve(CACHE_SIZE / 2);
}

float ChessEngine::evaluate(const ChessBoard& board) {
    uint64_t pos_key = board.hash();
    if (auto it = eval_cache.find(pos_key); it != eval_cache.end()) return it->second;

    float eval = raw_evaluate(board);
    eval_cache.emplace(pos_key, eval);
    return eval;
}

void ChessEngine::order_moves(const ChessBoard& board,
                              std::vector<ChessBoard::Move>& moves,
                              const ChessBoard::Move& tt_move) {
    if (moves.size() <= 1) return;

    constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900, 0, 0};
    bool has_tt_move = tt_move.internal_move != chess::Move::NO_MOVE;

    thread_local std::vector<std::pair<ChessBoard::Move, int>> scored;
    scored.clear();
    scored.reserve(moves.size());

    for (const auto& move : moves) {
        int score = 0;
        if (has_tt_move && move.internal_move == tt_move.internal_move) {
            score = 1000000;
        } else if (board.is_capture_move(move)) {
            int victim = PIECE_VALUES[board.piece_type_at(move.to())];
            int attacker = PIECE_VALUES[board.piece_type_at(move.from())];
            score = 100000 + victim * 10 - attacker;
        } else if (move.is_promotion()) {
            score = 90000;
        }
        scored.emplace_back(move, score);
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < moves.size(); ++i) moves[i] = scored[i].first;
}

SearchResult ChessEngine::get_best_move(const ChessBoard& board,
                                        const TimeControl& time_control) {
    auto legal_moves = board.get_legal_moves();

    if (legal_moves.empty()) {
        float score = board.is_in_check(board.turn()) ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }
    if (legal_moves.size() == 1) {
        return {legal_moves[0], 0.0f, 1, std::chrono::milliseconds(50), 1};
    }

    return iterative_deepening_search(board, calculate_search_time(time_control));
}

float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta,
                           bool is_pv) {
    if ((nodes_searched.fetch_add(1) & (TIME_CHECK_INTERVAL - 1)) == 0) check_time();
    if (should_stop.load()) return SEARCH_INTERRUPTED;

    if (board.is_stalemate() || board.is_draw()) {
        return 0.0f;
    }

    uint64_t pos_key = board.hash();
    ChessBoard::Move tt_move;

    if (auto it = transposition_table.find(pos_key);
        it != transposition_table.end() && it->second.depth >= depth) {
        const auto& entry = it->second;
        tt_move = entry.best_move;

        if (!is_pv) {
            switch (entry.type) {
                case TranspositionEntry::EXACT:
                    return entry.score;
                case TranspositionEntry::LOWER_BOUND:
                    if (entry.score >= beta) return entry.score;
                    alpha = std::max(alpha, entry.score);
                    break;
                case TranspositionEntry::UPPER_BOUND:
                    if (entry.score <= alpha) return entry.score;
                    beta = std::min(beta, entry.score);
                    break;
            }
            if (alpha >= beta) return entry.score;
        }
    }

    bool in_check = board.is_in_check(board.turn());

    // Check extension — search deeper when in check
    if (in_check && depth == 0) depth = 1;

    if (depth == 0) {
        return quiescence_search(board, alpha, beta, 0, in_check);
    }

    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        return in_check ? -MATE_VALUE : 0.0f;
    }

    // Null Move Pruning
    if (depth > config::search::null_move::MIN_DEPTH && !is_pv && !in_check &&
        beta < MATE_VALUE - config::search::null_move::MATE_MARGIN &&
        alpha > -MATE_VALUE + config::search::null_move::MATE_MARGIN) {
        ChessBoard null_board = board;
        null_board.board.makeNullMove();
        if (!null_board.is_game_over()) {
            int reduction = depth > config::search::null_move::DEEP_THRESHOLD
                                ? config::search::null_move::DEEP_REDUCTION
                                : config::search::null_move::SHALLOW_REDUCTION;
            int null_depth = depth - 1 - reduction;

            if (null_depth >= 0) {
                float null_score =
                    -negamax(null_board, null_depth, -beta, -beta + 1, false);
                if (null_score >= beta) {
                    return null_score;
                }
            }
        }
    }

    order_moves(board, legal_moves, tt_move);
    float best_score = -std::numeric_limits<float>::infinity();
    ChessBoard::Move best_move;
    auto node_type = TranspositionEntry::UPPER_BOUND;

    ChessBoard temp_board = board;
    int moves_searched = 0;
    for (const auto& move : legal_moves) {
        if (temp_board.make_move(move)) {
            moves_searched++;
            float score;

            // Late Move Reduction (LMR)
            if (moves_searched > config::search::lmr::MIN_MOVES_SEARCHED &&
                depth > config::search::lmr::MIN_DEPTH && !is_pv && !in_check &&
                !board.is_capture_move(move) && !move.is_promotion()) {
                int reduction =
                    moves_searched > config::search::lmr::MANY_MOVES_THRESHOLD
                        ? config::search::lmr::DEEP_REDUCTION
                        : config::search::lmr::SHALLOW_REDUCTION;
                int reduced_depth = depth - 1 - reduction;

                if (reduced_depth >= 0) {
                    score = -negamax(temp_board, reduced_depth, -beta, -alpha, false);

                    if (score > alpha) {
                        bool child_pv = is_pv && (moves_searched == 1);
                        score =
                            -negamax(temp_board, depth - 1, -beta, -alpha, child_pv);
                    }
                } else {
                    bool child_pv = is_pv && (moves_searched == 1);
                    score = -negamax(temp_board, depth - 1, -beta, -alpha, child_pv);
                }
            } else {
                bool child_pv = is_pv && (moves_searched == 1);
                score = -negamax(temp_board, depth - 1, -beta, -alpha, child_pv);
            }

            temp_board.unmake_move(move);

            if (score == SEARCH_INTERRUPTED) {
                return SEARCH_INTERRUPTED;
            }

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }

            if (score >= beta) {
                node_type = TranspositionEntry::LOWER_BOUND;
                break;
            }

            if (score > alpha) {
                alpha = score;
                node_type = TranspositionEntry::EXACT;
            }
        }
    }

    if (!tt_full) {
        transposition_table[pos_key] = {best_score, depth, node_type, best_move};
        if (transposition_table.size() >= CACHE_SIZE / 2) tt_full = true;
    }

    return best_score;
}

float ChessEngine::quiescence_search(const ChessBoard& board, float alpha, float beta,
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

    auto legal_moves = board.get_legal_moves();

    if (in_check && legal_moves.empty()) return -MATE_VALUE;

    // When in check, all moves must be searched; otherwise only tactical moves
    std::vector<ChessBoard::Move> tactical_moves;
    if (in_check) {
        tactical_moves = std::move(legal_moves);
    } else {
        for (const auto& move : legal_moves) {
            if (board.is_capture_move(move) || move.is_promotion()) {
                tactical_moves.push_back(move);
            }
        }
    }

    if (!tactical_moves.empty()) {
        order_moves(board, tactical_moves, ChessBoard::Move{});
        ChessBoard temp_board = board;

        for (const auto& move : tactical_moves) {
            if (temp_board.make_move(move)) {
                bool child_in_check = temp_board.is_in_check(temp_board.turn());
                float score = -quiescence_search(temp_board, -beta, -alpha,
                                                 qs_depth + 1, child_in_check);
                temp_board.unmake_move(move);

                if (score == SEARCH_INTERRUPTED) return SEARCH_INTERRUPTED;
                if (score >= beta) return beta;
                if (score > alpha) alpha = score;
            }
        }
    }

    return alpha;
}

void ChessEngine::check_time() {
    if (std::chrono::steady_clock::now() >= search_deadline) {
        should_stop.store(true);
    }
}

SearchResult ChessEngine::iterative_deepening_search(const ChessBoard& board,
                                                     int max_time_ms) {
    auto start_time = std::chrono::steady_clock::now();
    search_deadline = start_time + std::chrono::milliseconds(max_time_ms);
    should_stop.store(false);
    nodes_searched.store(0);
    eval_cache.clear();

    auto legal_moves = board.get_legal_moves();
    auto stm = board.turn();
    if (legal_moves.empty()) {
        float score = board.is_in_check(stm) ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }

    float static_eval = evaluate(board);
    static_eval = stm == ChessBoard::WHITE ? static_eval : -static_eval;
    SearchResult best_result{legal_moves[0], static_eval, 0, {}, 0};
    uint64_t pos_key = board.hash();

    for (int depth = 1; depth <= config::search::MAX_DEPTH; ++depth) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= max_time_ms) break;

        ChessBoard::Move tt_move;
        if (auto it = transposition_table.find(pos_key);
            it != transposition_table.end()) {
            tt_move = it->second.best_move;
        }

        order_moves(board, legal_moves, tt_move);
        ChessBoard::Move current_best_move = legal_moves[0];
        float current_best_score = -std::numeric_limits<float>::infinity();
        float alpha = -std::numeric_limits<float>::infinity();
        float beta = std::numeric_limits<float>::infinity();

        ChessBoard temp_board = board;
        bool completed_depth = true;
        int move_count = 0;

        for (const auto& move : legal_moves) {
            auto now_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time);
            if (now_elapsed.count() > max_time_ms || should_stop.load()) {
                completed_depth = false;
                break;
            }

            if (temp_board.make_move(move)) {
                move_count++;
                bool is_pv = (move_count == 1);

                float score = -negamax(temp_board, depth - 1, -beta, -alpha, is_pv);
                temp_board.unmake_move(move);

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
        }

        if (completed_depth && !should_stop.load()) {
            best_result = {current_best_move,
                           current_best_score,
                           depth,
                           {},
                           nodes_searched.load()};

            if (!tt_full) {
                transposition_table[pos_key] = {current_best_score, depth,
                                                TranspositionEntry::EXACT,
                                                current_best_move};
                if (transposition_table.size() >= CACHE_SIZE / 2) tt_full = true;
            }

            if (std::abs(current_best_score) > MATE_VALUE - 500) break;
        }

        if (!completed_depth) break;
    }

    best_result.time_used = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);

    return best_result;
}

void ChessEngine::clear_caches() {
    eval_cache.clear();
    transposition_table.clear();
    tt_full = false;
}
