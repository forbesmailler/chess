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
    switch (piece) {
        case ChessBoard::PAWN: return 100.0f;
        case ChessBoard::KNIGHT: return 320.0f;
        case ChessBoard::BISHOP: return 330.0f;
        case ChessBoard::ROOK: return 500.0f;
        case ChessBoard::QUEEN: return 900.0f;
        case ChessBoard::KING: return 20000.0f;
        default: return 0.0f;
    }
}

float ChessEngine::evaluate_position(const ChessBoard& board) {
    // Quick material evaluation for move ordering
    float material = 0.0f;
    
    for (int square = 0; square < 64; square++) {
        auto piece = board.piece_at(square);
        if (piece != ChessBoard::NONE) {
            float value = get_piece_value(static_cast<ChessBoard::PieceType>(piece & 7));
            if ((piece & 8) == 0) { // White piece
                material += value;
            } else { // Black piece
                material -= value;
            }
        }
    }
    
    return board.turn() == ChessBoard::WHITE ? material : -material;
}

float ChessEngine::evaluate(const ChessBoard& board) {
    // Terminal position detection with better scoring
    if (board.is_checkmate()) {
        return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
    }
    
    if (board.is_stalemate() || board.is_draw()) {
        return 0.0f;
    }
    
    // Check cache first
    std::string pos_key = get_position_key(board);
    auto cache_it = eval_cache.find(pos_key);
    if (cache_it != eval_cache.end()) {
        return cache_it->second;
    }
    
    // Use ML model for evaluation (output range: -1 to 1)
    auto features = FeatureExtractor::extract_features(board);
    auto proba = model->predict_proba(features);
    
    // Keep the evaluation in the -1 to 1 range from the ML model
    float eval = proba[2] - proba[0];
    
    // Cache the result
    if (eval_cache.size() >= CACHE_SIZE / 2) {
        clear_cache_if_needed();
    }
    eval_cache[pos_key] = eval;
    
    return eval;
}

int ChessEngine::score_move(const ChessBoard& board, const ChessBoard::Move& move) {
    int score = 0;
    
    // Most Valuable Victim - Least Valuable Attacker (MVV-LVA)
    if (move.is_capture()) {
        auto captured = board.piece_at(move.to());
        auto attacker = board.piece_at(move.from());
        
        if (captured != ChessBoard::NONE && attacker != ChessBoard::NONE) {
            score += static_cast<int>(get_piece_value(static_cast<ChessBoard::PieceType>(captured & 7))) * 10;
            score -= static_cast<int>(get_piece_value(static_cast<ChessBoard::PieceType>(attacker & 7)));
        }
    }
    
    // Promotion bonus
    if (move.is_promotion()) {
        score += 800; // High bonus for promotion
    }
    
    // Check bonus
    ChessBoard temp_board = board;
    if (temp_board.make_move(move)) {
        if (temp_board.is_in_check(temp_board.turn())) {
            score += 50;
        }
        temp_board.unmake_move(move);
    }
    
    // Center control bonus for non-captures
    if (!move.is_capture()) {
        int to_square = move.to();
        int file = to_square % 8;
        int rank = to_square / 8;
        
        // Bonus for moves to center squares
        if ((file >= 2 && file <= 5) && (rank >= 2 && rank <= 5)) {
            score += 10;
        }
    }
    
    return score;
}

std::vector<ChessBoard::Move> ChessEngine::order_moves(const ChessBoard& board, 
                                                      const std::vector<ChessBoard::Move>& moves, 
                                                      const ChessBoard::Move& tt_move) {
    std::vector<std::pair<ChessBoard::Move, int>> scored_moves;
    scored_moves.reserve(moves.size());
    
    for (const auto& move : moves) {
        int score = 0;
        
        // Transposition table move gets highest priority
        if (!tt_move.uci_string.empty() && move.uci() == tt_move.uci()) {
            score = 1000000; // Highest priority
        } else {
            score = score_move(board, move);
        }
        
        scored_moves.emplace_back(move, score);
    }
    
    // Sort by score (highest first)
    std::sort(scored_moves.begin(), scored_moves.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<ChessBoard::Move> ordered_moves;
    ordered_moves.reserve(moves.size());
    for (const auto& pair : scored_moves) {
        ordered_moves.push_back(pair.first);
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
    
    // Check transposition table for previous best move
    std::string pos_key = get_position_key(board);
    ChessBoard::Move tt_move;
    auto tt_it = transposition_table.find(pos_key);
    if (tt_it != transposition_table.end() && tt_it->second.depth >= search_depth) {
        tt_move = tt_it->second.best_move;
    }
    
    // Order moves for better alpha-beta pruning
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
            if (alpha >= beta) {
                break; // Alpha-beta pruning
            }
        }
    }
    
    // Store in transposition table
    TranspositionEntry entry;
    entry.score = best_score;
    entry.depth = search_depth;
    entry.type = TranspositionEntry::EXACT;
    entry.best_move = best_move;
    
    if (transposition_table.size() < CACHE_SIZE / 2) {
        transposition_table[pos_key] = entry;
    }
    
    return best_move;
}
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
    
float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv) {
    // Check for mate/draw
    if (board.is_checkmate()) {
        return -MATE_VALUE + (search_depth - depth); // Prefer quicker mates
    }
    
    if (board.is_stalemate() || board.is_draw()) {
        return 0.0f;
    }
    
    // Transposition table lookup
    std::string pos_key = get_position_key(board);
    auto tt_it = transposition_table.find(pos_key);
    ChessBoard::Move tt_move;
    
    if (tt_it != transposition_table.end() && tt_it->second.depth >= depth) {
        const auto& entry = tt_it->second;
        tt_move = entry.best_move;
        
        // Use stored score if not in PV and conditions are met
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
            
            if (alpha >= beta) {
                return entry.score;
            }
        }
    }
    
    // Quiescence search at leaf nodes
    if (depth == 0) {
        return quiescence_search(board, alpha, beta);
    }
    
    auto legal_moves = board.get_legal_moves();
    if (legal_moves.empty()) {
        return board.is_in_check(board.turn()) ? -MATE_VALUE + (search_depth - depth) : 0.0f;
    }
    
    // Move ordering
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
                // Beta cutoff
                node_type = TranspositionEntry::LOWER_BOUND;
                break;
            }
            
            if (score > alpha) {
                alpha = score;
                node_type = TranspositionEntry::EXACT;
            }
        }
    }
    
    // Store in transposition table
    if (transposition_table.size() < CACHE_SIZE / 2) {
        TranspositionEntry entry;
        entry.score = best_score;
        entry.depth = depth;
        entry.type = node_type;
        entry.best_move = best_move;
        transposition_table[pos_key] = entry;
    }
    
    return best_score;
}

float ChessEngine::quiescence_search(const ChessBoard& board, float alpha, float beta, int qs_depth) {
    // Limit quiescence search depth to prevent stack overflow
    if (qs_depth >= 10) {
        return evaluate(board);
    }
    
    // Standing pat evaluation
    float stand_pat = evaluate(board);
    
    if (stand_pat >= beta) {
        return beta;
    }
    
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }
    
    // Only search captures and checks in quiescence
    auto legal_moves = board.get_legal_moves();
    std::vector<ChessBoard::Move> tactical_moves;
    
    ChessBoard temp_board = board;
    for (const auto& move : legal_moves) {
        if (move.is_capture() || move.is_promotion()) {
            tactical_moves.push_back(move);
        } else {
            // Check if move gives check
            if (temp_board.make_move(move)) {
                if (temp_board.is_in_check(temp_board.turn())) {
                    tactical_moves.push_back(move);
                }
                temp_board.unmake_move(move);
            }
        }
    }
    
    // Order tactical moves
    auto ordered_tactical = order_moves(board, tactical_moves);
    
    for (const auto& move : ordered_tactical) {
        if (temp_board.make_move(move)) {
            float score = -quiescence_search(temp_board, -beta, -alpha, qs_depth + 1);
            temp_board.unmake_move(move);
            
            if (score >= beta) {
                return beta; // Beta cutoff
            }
            
            if (score > alpha) {
                alpha = score;
            }
        }
    }
    
std::string ChessEngine::get_position_key(const ChessBoard& board) const {
    // Use a simplified position key for better performance
    std::string fen = board.to_fen();
    size_t space_pos = fen.find(' ');
    if (space_pos != std::string::npos) {
        size_t second_space = fen.find(' ', space_pos + 1);
        if (second_space != std::string::npos) {
            return fen.substr(0, second_space); // Include piece positions and turn
        }
    }
    return fen;
}

void ChessEngine::clear_cache_if_needed() {
    // Clear oldest half of entries when cache gets too large
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
