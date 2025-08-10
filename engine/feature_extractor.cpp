#include "feature_extractor.h"
#include "chess_board.h"
#include <algorithm>
#include <sstream>
#include <cctype>

std::vector<float> FeatureExtractor::extract_features(const std::string& fen) {
    return extract_features(ChessBoard(fen));
}

std::vector<float> FeatureExtractor::extract_features(const ChessBoard& board) {
    auto piece_features = extract_piece_features(board);
    auto additional_features = extract_additional_features(board);
    
    std::vector<float> base_features;
    base_features.reserve(776);
    
    for (float f : piece_features) base_features.push_back(f);
    for (float f : additional_features) base_features.push_back(f);
    
    float factor = static_cast<float>(board.piece_count() - 2) / 30.0f;
    
    std::vector<float> features;
    features.reserve(FEATURE_SIZE);
    
    for (float f : base_features) features.push_back(f * factor);
    for (float f : base_features) features.push_back(f * (1.0f - factor));
    
    return features;
}

std::array<float, 768> FeatureExtractor::extract_piece_features(const ChessBoard& board) {
    std::array<float, 768> piece_arr;
    piece_arr.fill(0.0f);
    
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str;
    iss >> board_str;
    
    int square = 56;
    
    for (char c : board_str) {
        if (c == '/') {
            square -= 16;
        } else if (std::isdigit(c)) {
            square += (c - '0');
        } else {
            static const int piece_map[] = {0, 1, 2, 3, 4, 5, 6}; // p=1, n=2, b=3, r=4, q=5, k=6
            int piece_type = 0;
            switch (std::tolower(c)) {
                case 'p': piece_type = 1; break;
                case 'n': piece_type = 2; break;
                case 'b': piece_type = 3; break;
                case 'r': piece_type = 4; break;
                case 'q': piece_type = 5; break;
                case 'k': piece_type = 6; break;
            }
            
            if (piece_type > 0) {
                int idx = (piece_type - 1) + (std::isupper(c) ? 0 : 6);
                piece_arr[idx * 64 + square] = 1.0f;
            }
            square++;
        }
    }
    
    return piece_arr;
}

std::array<float, 8> FeatureExtractor::extract_additional_features(const ChessBoard& board) {
    std::array<float, 8> features;
    features.fill(0.0f);
    
    // Get current player's features
    auto moves = board.get_legal_moves();
    float move_count = static_cast<float>(moves.size());
    float capture_count = 0.0f;
    float check_count = 0.0f;
    
    // Create a mutable copy for testing moves
    ChessBoard temp_board = board;
    
    for (const auto& move : moves) {
        if (temp_board.is_capture_move(move)) {
            capture_count += 1.0f;
        }
        
        // Test if move gives check
        if (temp_board.make_move(move)) {
            ChessBoard::Color opponent_color = (temp_board.turn() == ChessBoard::WHITE) ? 
                                             ChessBoard::BLACK : ChessBoard::WHITE;
            if (temp_board.is_in_check(opponent_color)) {
                check_count += 1.0f;
            }
            temp_board.unmake_move(move);
        }
    }
    
    // Get current check status
    float in_check = board.is_in_check(board.turn()) ? 1.0f : 0.0f;
    
    // Create a board with flipped turn to get opponent features
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Flip the turn
    turn_str = (turn_str == "w") ? "b" : "w";
    std::string flipped_fen = board_str + " " + turn_str + " " + castling_str + " " + 
                             ep_str + " " + halfmove_str + " " + fullmove_str;
    
    ChessBoard flipped_board(flipped_fen);
    
    // Get opponent features (if the position is legal with flipped turn)
    float opponent_move_count = 0.0f;
    float opponent_capture_count = 0.0f;
    float opponent_check_count = 0.0f;
    float opponent_in_check = 0.0f;
    
    if (!flipped_board.is_game_over()) {
        auto opponent_moves = flipped_board.get_legal_moves();
        opponent_move_count = static_cast<float>(opponent_moves.size());
        
        ChessBoard temp_flipped = flipped_board;
        for (const auto& move : opponent_moves) {
            if (temp_flipped.is_capture_move(move)) {
                opponent_capture_count += 1.0f;
            }
            
            if (temp_flipped.make_move(move)) {
                ChessBoard::Color target_color = (temp_flipped.turn() == ChessBoard::WHITE) ? 
                                               ChessBoard::BLACK : ChessBoard::WHITE;
                if (temp_flipped.is_in_check(target_color)) {
                    opponent_check_count += 1.0f;
                }
                temp_flipped.unmake_move(move);
            }
        }
        
        opponent_in_check = flipped_board.is_in_check(flipped_board.turn()) ? 1.0f : 0.0f;
    }
    
    // Assign features based on whose turn it is in the original position
    if (board.turn() == ChessBoard::WHITE) {
        features[0] = move_count;              // white moves
        features[1] = opponent_move_count;     // black moves  
        features[2] = capture_count;           // white captures
        features[3] = opponent_capture_count;  // black captures
        features[4] = check_count;             // white checks
        features[5] = opponent_check_count;    // black checks
        features[6] = in_check;                // white in check
        features[7] = opponent_in_check;       // black in check
    } else {
        features[0] = opponent_move_count;     // white moves
        features[1] = move_count;              // black moves
        features[2] = opponent_capture_count;  // white captures
        features[3] = capture_count;           // black captures
        features[4] = opponent_check_count;    // white checks
        features[5] = check_count;             // black checks
        features[6] = opponent_in_check;       // white in check
        features[7] = in_check;                // black in check
    }
    
    return features;
}
