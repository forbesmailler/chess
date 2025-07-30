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

float ChessEngine::evaluate(const ChessBoard& board) {
    // NOTE: Evaluation is always from WHITE's perspective
    // The caller is responsible for flipping the sign for Black
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
    
    float raw_eval = proba[2] - proba[0];  // Range: -1 to 1
    float eval = raw_eval * 10000.0f;  // Scale by 10,000
    
    // Endgame detection and bonus for forcing checkmate
    int piece_count = board.piece_count();
    if (piece_count <= 10) {  // Endgame
        // In endgame, add bonus for positions that restrict opponent's king
        // This helps the bot play more forcing moves toward mate
        auto legal_moves = board.get_legal_moves();
        if (legal_moves.size() < 10) {  // Current player has few moves (being restricted)
            float restriction_penalty = (10 - legal_moves.size()) * 50.0f;
            // Apply penalty/bonus from WHITE's perspective
            if (board.turn() == ChessBoard::WHITE) {
                eval -= restriction_penalty; // White is restricted (bad for White)
            } else {
                eval += restriction_penalty; // Black is restricted (good for White)
            }
        }
        
        // Extra evaluation for checks in endgame
        if (board.is_in_check(board.turn())) {
            // Being in check is bad for the current player
            if (board.turn() == ChessBoard::WHITE) {
                eval -= 300.0f; // White in check (bad for White)
            } else {
                eval += 300.0f; // Black in check (good for White)
            }
        }
    }
    
    // Cache the result
    if (eval_cache.size() >= CACHE_SIZE / 2) {
        clear_cache_if_needed();
    }
    eval_cache[pos_key] = eval;
    
    return eval;
}

int ChessEngine::score_move(const ChessBoard& board, const ChessBoard::Move& move) {
    // Use ML model to score moves by evaluating the resulting position
    ChessBoard temp_board = board;
    
    if (!temp_board.make_move(move)) {
        return -MATE_VALUE; // Invalid move gets very low score
    }
    
    // Get ML evaluation of the position after the move
    float eval_after = evaluate(temp_board);
    
    // Convert from White's perspective to current player's perspective
    float score_for_current_player = board.turn() == ChessBoard::WHITE ? eval_after : -eval_after;
    
    temp_board.unmake_move(move);
    
    // Convert to int for move ordering (the function signature requires int)
    // Scale down since this is just for ordering moves, not final evaluation
    int move_score = static_cast<int>(score_for_current_player / 10.0f);
    
    return move_score;
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
    
    return ordered_moves;
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

float ChessEngine::negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv) {
    if (board.is_checkmate()) {
        // Return mate score adjusted for distance (prefer quicker mates)
        return -MATE_VALUE + (search_depth - depth);
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
        // No legal moves - check if it's mate or stalemate
        if (board.is_in_check(board.turn())) {
            return -MATE_VALUE + (search_depth - depth); // Mate (prefer quicker)
        } else {
            return 0.0f; // Stalemate
        }
    }
    
    // In endgame with few pieces, extend search depth to find mates
    int piece_count = board.piece_count();
    bool extend_search = false;
    if (piece_count <= 10 && depth == 0) {
        extend_search = true;
        depth = 1; // Extend search by 1 ply in endgame
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
    
    // Store in transposition table (don't store extended search results)
    if (!extend_search && transposition_table.size() < CACHE_SIZE / 2) {
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
        float eval = evaluate(board);
        // Convert from White's perspective to current player's perspective
        return board.turn() == ChessBoard::WHITE ? eval : -eval;
    }
    
    // Standing pat evaluation
    float stand_pat = evaluate(board);
    // Convert from White's perspective to current player's perspective
    stand_pat = board.turn() == ChessBoard::WHITE ? stand_pat : -stand_pat;
    
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
        if (board.is_capture_move(move) || move.is_promotion()) {
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
    
    return alpha; // Return the best score found
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
