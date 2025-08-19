#include "chess_engine.h"
#include "feature_extractor.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <sstream>

ChessEngine::ChessEngine(std::shared_ptr<LogisticModel> model, int max_time_ms) 
    : model(model), max_search_time_ms(max_time_ms) {
    eval_cache.reserve(CACHE_SIZE / 2);
    transposition_table.reserve(CACHE_SIZE / 2);
}

float ChessEngine::evaluate(const ChessBoard& board) {
    if (board.is_checkmate()) {
        return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
    }
    
    if (board.is_stalemate() || board.is_draw()) return 0.0f;
    
    std::string pos_key = get_position_key(board);
    auto cache_it = eval_cache.find(pos_key);
    if (cache_it != eval_cache.end()) return cache_it->second;
    
    auto features = FeatureExtractor::extract_features(board);
    auto proba = model->predict_proba(features);
    float eval = (proba[2] - proba[0]) * MATE_VALUE;
    
    if (eval_cache.size() >= CACHE_SIZE / 2) clear_cache_if_needed();
    eval_cache[pos_key] = eval;
    return eval;
}

int ChessEngine::score_move(const ChessBoard& board, const ChessBoard::Move& move) {
    ChessBoard temp_board = board;
    if (!temp_board.make_move(move)) return -MATE_VALUE;
    
    float eval_after = evaluate(temp_board);
    float score_for_current_player = board.turn() == ChessBoard::WHITE ? eval_after : -eval_after;
    temp_board.unmake_move(move);

    return static_cast<int>(score_for_current_player / 10.0f);
}

std::vector<ChessBoard::Move> ChessEngine::order_moves(const ChessBoard& board, 
                                                      const std::vector<ChessBoard::Move>& moves, 
                                                      const ChessBoard::Move& tt_move) {
    if (moves.size() <= 1) return moves;
    
    std::vector<std::pair<ChessBoard::Move, int>> scored_moves;
    scored_moves.reserve(moves.size());
    
    for (const auto& move : moves) {
        int score = (!tt_move.uci_string.empty() && move.uci() == tt_move.uci()) ? 
                    1000000 : score_move(board, move);
        scored_moves.emplace_back(move, score);
    }
    
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<ChessBoard::Move> ordered_moves;
    ordered_moves.reserve(moves.size());
    for (const auto& pair : scored_moves) {
        ordered_moves.push_back(pair.first);
    }
    
    return ordered_moves;
}

SearchResult ChessEngine::get_best_move(const ChessBoard& board, const TimeControl& time_control) {
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        // No legal moves - check if it's checkmate or stalemate
        if (board.is_in_check(board.turn())) {
            return {ChessBoard::Move{}, -MATE_VALUE, 0, std::chrono::milliseconds(0), 0};
        } else {
            return {ChessBoard::Move{}, 0.0f, 0, std::chrono::milliseconds(0), 0}; // Stalemate
        }
    }
    if (legal_moves.size() == 1) {
        return {legal_moves[0], 0.0f, 1, std::chrono::milliseconds(50), 1};
    }
    
    int search_time_ms = calculate_search_time(time_control);
    
    return iterative_deepening_search(board, search_time_ms);
}

float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv) {
    // Check if search should stop
    if (should_stop.load()) return 0.0f;
    
    // Increment node counter
    nodes_searched.fetch_add(1);
    
    // Periodically check if we should stop (every 1000 nodes to avoid overhead)
    if (nodes_searched.load() % 1000 == 0 && should_stop.load()) {
        return 0.0f;
    }
    
    if (board.is_stalemate() || board.is_draw()) return 0.0f;
    
    std::string pos_key = get_position_key(board);
    auto tt_it = transposition_table.find(pos_key);
    ChessBoard::Move tt_move;
    
    if (tt_it != transposition_table.end() && tt_it->second.depth >= depth) {
        const auto& entry = tt_it->second;
        tt_move = entry.best_move;
        
        if (!is_pv) {
            switch (entry.type) {
                case TranspositionEntry::EXACT: return entry.score;
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
    
    if (depth == 0) return quiescence_search(board, alpha, beta);
    
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        return board.is_in_check(board.turn()) ? -MATE_VALUE : 0.0f;
    }
    
    // Check extension - but only at low depths
    if (board.is_in_check(board.turn()) && depth == 0) {
        depth = 1;
    }
    
    auto ordered_moves = order_moves(board, legal_moves, tt_move);
    float best_score = -std::numeric_limits<float>::infinity();
    ChessBoard::Move best_move;
    TranspositionEntry::NodeType node_type = TranspositionEntry::UPPER_BOUND;
    
    ChessBoard temp_board = board;
    for (const auto& move : ordered_moves) {
        if (temp_board.make_move(move)) {
            // Simple negamax search without LMR/PVS complications
            float score = -negamax(temp_board, depth - 1, -beta, -alpha, is_pv && best_move.uci_string.empty());
            temp_board.unmake_move(move);
            
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
    
    if (transposition_table.size() < CACHE_SIZE / 2) {
        transposition_table[pos_key] = {best_score, depth, node_type, best_move};
    }
    
    return best_score;
}

float ChessEngine::quiescence_search(const ChessBoard& board, float alpha, float beta, int qs_depth) {
    // Fixed max depth for quiescence search
    int max_qs_depth = 8;  // Reduced for better performance
    
    if (qs_depth >= max_qs_depth) {
        float eval = evaluate(board);
        return board.turn() == ChessBoard::WHITE ? eval : -eval;
    }
    
    float stand_pat = evaluate(board);
    stand_pat = board.turn() == ChessBoard::WHITE ? stand_pat : -stand_pat;
    
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    
    auto legal_moves = board.get_legal_moves();
    std::vector<ChessBoard::Move> tactical_moves;
    tactical_moves.reserve(legal_moves.size() / 4);  // Estimate
    
    // Look at captures, promotions, and checks
    ChessBoard temp_board = board;
    for (const auto& move : legal_moves) {
        if (board.is_capture_move(move) || move.is_promotion()) {
            tactical_moves.push_back(move);
        } else if (temp_board.make_move(move)) {
            // Check if move gives check
            if (temp_board.is_in_check(temp_board.turn())) {
                tactical_moves.push_back(move);
            }
            temp_board.unmake_move(move);
        }
    }
    
    // Simple ordering: captures/promotions/checks
    temp_board = board;
    for (const auto& move : tactical_moves) {
        if (temp_board.make_move(move)) {
            float score = -quiescence_search(temp_board, -beta, -alpha, qs_depth + 1);
            temp_board.unmake_move(move);
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
    }
    
    return alpha;
}

std::string ChessEngine::get_position_key(const ChessBoard& board) const {
    std::string fen = board.to_fen();
    // Find second space to truncate after castling rights
    size_t first_space = fen.find(' ');
    if (first_space != std::string::npos) {
        size_t second_space = fen.find(' ', first_space + 1);
        if (second_space != std::string::npos) {
            size_t third_space = fen.find(' ', second_space + 1);
            if (third_space != std::string::npos) {
                return fen.substr(0, third_space);  // Include en passant square
            }
        }
    }
    return fen;
}

void ChessEngine::clear_cache_if_needed() {
    // Use a simple random eviction strategy for better performance
    if (eval_cache.size() >= CACHE_SIZE / 2) {
        auto it = eval_cache.begin();
        std::advance(it, eval_cache.size() / 2);  // Remove half
        eval_cache.erase(eval_cache.begin(), it);
    }
    
    if (transposition_table.size() >= CACHE_SIZE / 2) {
        auto it = transposition_table.begin();
        std::advance(it, transposition_table.size() / 2);  // Remove half  
        transposition_table.erase(transposition_table.begin(), it);
    }
}

int ChessEngine::calculate_search_time(const TimeControl& time_control) {
    if (time_control.time_left_ms <= 0) {
        return max_search_time_ms; // Default minimum time if no time control
    }
    
    // Simple time management: increment + (remaining time / 40), capped at max_search_time_ms
    int base_time = time_control.time_left_ms;
    int increment = time_control.increment_ms;
    
    int allocated_time = increment + (base_time / 40);
    
    // Cap at configured max search time
    allocated_time = std::min(allocated_time, max_search_time_ms);
    
    return allocated_time;
}

SearchResult ChessEngine::iterative_deepening_search(const ChessBoard& board, int max_time_ms) {
    auto start_time = std::chrono::steady_clock::now();
    should_stop.store(false);
    nodes_searched.store(0);
    
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        // No legal moves - check if it's checkmate or stalemate
        if (board.is_in_check(board.turn())) {
            return {ChessBoard::Move{}, -MATE_VALUE, 0, std::chrono::milliseconds(0), 0};
        } else {
            return {ChessBoard::Move{}, 0.0f, 0, std::chrono::milliseconds(0), 0}; // Stalemate
        }
    }
    
    SearchResult best_result;
    best_result.best_move = legal_moves[0]; // Fallback move
    best_result.score = -std::numeric_limits<float>::infinity();
    best_result.depth_reached = 0;
    best_result.nodes_searched = 0;
    
    // Iterative deepening loop - continue until time runs out
    for (int depth = 1; depth <= 50; ++depth) { // Cap at reasonable depth of 50
        auto iteration_start = std::chrono::steady_clock::now();
        
        // Check if we have enough time for this depth
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_start - start_time);
        if (elapsed.count() >= max_time_ms) {
            break; // Time is up
        }

        std::string pos_key = get_position_key(board);
        ChessBoard::Move tt_move;
        auto tt_it = transposition_table.find(pos_key);
        if (tt_it != transposition_table.end()) {
            tt_move = tt_it->second.best_move;
        }
        
        auto ordered_moves = order_moves(board, legal_moves, tt_move);
        ChessBoard::Move current_best_move = ordered_moves[0];
        float current_best_score = -std::numeric_limits<float>::infinity();
        float alpha = -std::numeric_limits<float>::infinity();
        float beta = std::numeric_limits<float>::infinity();
        
        ChessBoard temp_board = board;
        bool completed_depth = true;
        
        for (const auto& move : ordered_moves) {
            // Check time before each move
            auto now = std::chrono::steady_clock::now();
            auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
            if (total_elapsed.count() > max_time_ms || should_stop.load()) {
                completed_depth = false;
                break;
            }
            
            if (temp_board.make_move(move)) {
                float score = -negamax(temp_board, depth - 1, -beta, -alpha, true);
                temp_board.unmake_move(move);
                
                // Check if search was stopped
                if (should_stop.load()) {
                    completed_depth = false;
                    break;
                }
                
                if (score > current_best_score) {
                    current_best_score = score;
                    current_best_move = move;
                }
                
                alpha = std::max(alpha, score);
                if (alpha >= beta) break; // Alpha-beta cutoff
            }
        }
        
        // Only update result if we completed this depth
        if (completed_depth && !should_stop.load()) {
            best_result.best_move = current_best_move;
            best_result.score = current_best_score;
            best_result.depth_reached = depth;
            
            // Update transposition table
            if (transposition_table.size() < CACHE_SIZE / 2) {
                transposition_table[pos_key] = {current_best_score, depth, TranspositionEntry::EXACT, current_best_move};
            }
        } else {
            // Didn't complete this depth, stop searching
            break;
        }
        
        if (std::abs(current_best_score) == MATE_VALUE) {
            break;
        }
        
        // Time management: if this depth took a long time, probably won't have time for next depth
        auto iteration_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - iteration_start);
        auto total_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    best_result.time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    best_result.nodes_searched = nodes_searched.load();
    
    return best_result;
}
