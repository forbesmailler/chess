#include "chess_engine.h"
#include "feature_extractor.h"
#include <algorithm>
#include <limits>
#include <cmath>

ChessEngine::ChessEngine(std::shared_ptr<LogisticModel> model, int search_depth) 
    : model(model), search_depth(search_depth) {
    eval_cache.reserve(CACHE_SIZE / 2);
    transposition_table.reserve(CACHE_SIZE / 2);
}

float ChessEngine::get_piece_value(ChessBoard::PieceType piece) const {
    static const float values[] = {100.0f, 320.0f, 330.0f, 500.0f, 900.0f, 20000.0f, 0.0f};
    return values[piece];
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
    float eval = (proba[2] - proba[0]) * 10000.0f;
    
    int piece_count = board.piece_count();
    if (piece_count <= 10) {
        auto legal_moves = board.get_legal_moves();
        if (legal_moves.size() < 10) {
            float restriction_penalty = (10 - legal_moves.size()) * 50.0f;
            eval += (board.turn() == ChessBoard::WHITE) ? -restriction_penalty : restriction_penalty;
        }
        
        if (board.is_in_check(board.turn())) {
            eval += (board.turn() == ChessBoard::WHITE) ? -300.0f : 300.0f;
        }
    }
    
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

ChessBoard::Move ChessEngine::get_best_move(const ChessBoard& board) {
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) return ChessBoard::Move{};
    if (legal_moves.size() == 1) return legal_moves[0];
    
    std::string pos_key = get_position_key(board);
    ChessBoard::Move tt_move;
    auto tt_it = transposition_table.find(pos_key);
    if (tt_it != transposition_table.end() && tt_it->second.depth >= search_depth) {
        tt_move = tt_it->second.best_move;
    }
    
    auto ordered_moves = order_moves(board, legal_moves, tt_move);
    ChessBoard::Move best_move = ordered_moves[0];
    float best_score = -std::numeric_limits<float>::infinity();
    float alpha = -std::numeric_limits<float>::infinity();
    float beta = std::numeric_limits<float>::infinity();
    
    ChessBoard temp_board = board;
    for (const auto& move : ordered_moves) {
        if (temp_board.make_move(move)) {
            float score = -negamax(temp_board, search_depth - 1, -beta, -alpha, true);
            temp_board.unmake_move(move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
    }
    
    if (transposition_table.size() < CACHE_SIZE / 2) {
        transposition_table[pos_key] = {best_score, search_depth, TranspositionEntry::EXACT, best_move};
    }
    
    return best_move;
}

float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv) {
    if (board.is_checkmate()) return -MATE_VALUE + (search_depth - depth);
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
        return board.is_in_check(board.turn()) ? -MATE_VALUE + (search_depth - depth) : 0.0f;
    }
    
    bool extend_search = (board.piece_count() <= 10 && depth == 0);
    if (extend_search) depth = 1;
    
    auto ordered_moves = order_moves(board, legal_moves, tt_move);
    float best_score = -std::numeric_limits<float>::infinity();
    ChessBoard::Move best_move;
    TranspositionEntry::NodeType node_type = TranspositionEntry::UPPER_BOUND;
    
    ChessBoard temp_board = board;
    for (const auto& move : ordered_moves) {
        if (temp_board.make_move(move)) {
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
    
    if (!extend_search && transposition_table.size() < CACHE_SIZE / 2) {
        transposition_table[pos_key] = {best_score, depth, node_type, best_move};
    }
    
    return best_score;
}

float ChessEngine::quiescence_search(const ChessBoard& board, float alpha, float beta, int qs_depth) {
    if (qs_depth >= 10) {
        float eval = evaluate(board);
        return board.turn() == ChessBoard::WHITE ? eval : -eval;
    }
    
    float stand_pat = evaluate(board);
    stand_pat = board.turn() == ChessBoard::WHITE ? stand_pat : -stand_pat;
    
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    
    auto legal_moves = board.get_legal_moves();
    std::vector<ChessBoard::Move> tactical_moves;
    
    ChessBoard temp_board = board;
    for (const auto& move : legal_moves) {
        if (board.is_capture_move(move) || move.is_promotion()) {
            tactical_moves.push_back(move);
        } else if (temp_board.make_move(move)) {
            if (temp_board.is_in_check(temp_board.turn())) {
                tactical_moves.push_back(move);
            }
            temp_board.unmake_move(move);
        }
    }
    
    auto ordered_tactical = order_moves(board, tactical_moves);
    for (const auto& move : ordered_tactical) {
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
    size_t space_pos = fen.find(' ');
    if (space_pos != std::string::npos) {
        size_t second_space = fen.find(' ', space_pos + 1);
        if (second_space != std::string::npos) {
            return fen.substr(0, second_space);
        }
    }
    return fen;
}

void ChessEngine::clear_cache_if_needed() {
    if (eval_cache.size() >= CACHE_SIZE / 2) {
        auto it = eval_cache.begin();
        std::advance(it, eval_cache.size() / 2);
        eval_cache.erase(eval_cache.begin(), it);
    }
    
    if (transposition_table.size() >= CACHE_SIZE / 2) {
        auto it = transposition_table.begin();
        std::advance(it, transposition_table.size() / 2);
        transposition_table.erase(transposition_table.begin(), it);
    }
}
