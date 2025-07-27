#include "chess_engine.h"
#include "feature_extractor.h"
#include <algorithm>
#include <limits>

ChessEngine::ChessEngine(std::shared_ptr<LogisticModel> model, int search_depth) 
    : model(model), search_depth(search_depth) {
    eval_cache.reserve(CACHE_SIZE); // Pre-allocate cache
}

float ChessEngine::evaluate(const ChessBoard& board) {
    // Terminal position detection
    if (board.is_checkmate()) {
        return board.turn() == ChessBoard::WHITE ? -WIN_VALUE : WIN_VALUE;
    }
    
    if (board.is_stalemate() || board.is_draw()) {
        return 0.0f;
    }
    
    // Check cache first
    std::string fen = board.to_fen();
    auto cache_it = eval_cache.find(fen);
    if (cache_it != eval_cache.end()) {
        return cache_it->second;
    }
    
    // Probability-based evaluation
    auto features = FeatureExtractor::extract_features(board);
    auto proba = model->predict_proba(features);
    
    // proba[0] = white win, proba[1] = draw, proba[2] = black win
    float eval = proba[2] - proba[0];
    
    // Cache the result
    if (eval_cache.size() >= CACHE_SIZE) {
        clear_cache_if_needed();
    }
    eval_cache[fen] = eval;
    
    return eval;
}

ChessBoard::Move ChessEngine::get_best_move(const ChessBoard& board) {
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        return ChessBoard::Move{}; // No legal moves
    }
    
    // Single move optimization
    if (legal_moves.size() == 1) {
        return legal_moves[0];
    }
    
    ChessBoard::Move best_move = legal_moves[0];
    float best_score = -std::numeric_limits<float>::infinity();
    float alpha = -std::numeric_limits<float>::infinity();
    float beta = std::numeric_limits<float>::infinity();
    
    ChessBoard temp_board = board;
    
    for (const auto& move : legal_moves) {
        if (temp_board.make_move(move)) {
            float score = -negamax(temp_board, search_depth - 1, -beta, -alpha);
            temp_board.unmake_move(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                break; // Alpha-beta pruning
            }
        }
    }
    
    return best_move;
}

float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta) {
    // Terminal position check first (fastest)
    if (board.is_checkmate()) {
        return -WIN_VALUE * (depth + 1); // Prefer faster mate
    }
    
    if (board.is_stalemate() || board.is_draw()) {
        return 0.0f;
    }
    
    if (depth == 0) {
        float val = evaluate(board);
        return board.turn() == ChessBoard::WHITE ? val : -val;
    }
    
    // Check cache with simplified key
    std::string cache_key = board.to_fen().substr(0, board.to_fen().find(' ', 50)) + std::to_string(depth);
    auto cache_it = eval_cache.find(cache_key);
    if (cache_it != eval_cache.end()) {
        return cache_it->second;
    }
    
    float value = -std::numeric_limits<float>::infinity();
    auto legal_moves = board.get_legal_moves();
    
    // Early termination if no moves
    if (legal_moves.empty()) {
        return board.is_in_check(board.turn()) ? -WIN_VALUE * (depth + 1) : 0.0f;
    }
    
    ChessBoard temp_board = board;
    
    for (const auto& move : legal_moves) {
        if (temp_board.make_move(move)) {
            float score = -negamax(temp_board, depth - 1, -beta, -alpha);
            temp_board.unmake_move(move);
            
            value = std::max(value, score);
            alpha = std::max(alpha, score);
            
            if (alpha >= beta) {
                break; // Alpha-beta pruning
            }
        }
    }
    
    // Cache the result with size check
    if (eval_cache.size() < CACHE_SIZE) {
        eval_cache[cache_key] = value;
    } else {
        clear_cache_if_needed();
    }
    
    return value;
}

void ChessEngine::clear_cache_if_needed() {
    if (eval_cache.size() >= CACHE_SIZE) {
        eval_cache.clear();
    }
}
