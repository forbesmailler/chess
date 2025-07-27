#include "chess_engine.h"
#include "feature_extractor.h"
#include <algorithm>
#include <limits>

ChessEngine::ChessEngine(std::shared_ptr<LogisticModel> model) : model(model) {}

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
    
    ChessBoard::Move best_move = legal_moves[0];
    float best_score = -std::numeric_limits<float>::infinity();
    float alpha = -std::numeric_limits<float>::infinity();
    float beta = std::numeric_limits<float>::infinity();
    
    ChessBoard temp_board = board;
    
    for (const auto& move : legal_moves) {
        if (temp_board.make_move(move)) {
            float score = -negamax(temp_board, DEFAULT_DEPTH - 1, -beta, -alpha);
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
    if (depth == 0 || board.is_game_over()) {
        float val = evaluate(board);
        return board.turn() == ChessBoard::WHITE ? val : -val;
    }
    
    // Check cache
    std::string fen = board.to_fen() + "_" + std::to_string(depth);
    auto cache_it = eval_cache.find(fen);
    if (cache_it != eval_cache.end()) {
        return cache_it->second;
    }
    
    float value = -std::numeric_limits<float>::infinity();
    auto legal_moves = board.get_legal_moves();
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
    
    // Cache the result
    if (eval_cache.size() < CACHE_SIZE) {
        eval_cache[fen] = value;
    }
    
    return value;
}

void ChessEngine::clear_cache_if_needed() {
    if (eval_cache.size() >= CACHE_SIZE) {
        eval_cache.clear();
    }
}
