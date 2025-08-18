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
    base_features.reserve(770);
    
    for (float f : piece_features) base_features.push_back(f);
    for (float f : additional_features) base_features.push_back(f);
    
    float factor = static_cast<float>(board.piece_count() - 2) / 30.0f;
    
    std::vector<float> features;
    features.reserve(FEATURE_SIZE);
    
    // Add scaled features (770 * 2 = 1540)
    for (float f : base_features) features.push_back(f * factor);
    for (float f : base_features) features.push_back(f * (1.0f - factor));
    
    // Add mobility features separately
    auto mobility_features = extract_mobility_features(board);
    features.push_back(mobility_features[0]);  // white mobility
    features.push_back(mobility_features[1]);  // black mobility
    
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

std::array<float, 2> FeatureExtractor::extract_additional_features(const ChessBoard& board) {
    std::array<float, 2> features;
    features.fill(0.0f);
    
    // 1. Is white in check?
    // 2. Is black in check?
    float white_in_check = 0.0f;
    float black_in_check = 0.0f;
    
    if (board.turn() == ChessBoard::WHITE) {
        white_in_check = board.is_in_check(ChessBoard::WHITE) ? 1.0f : 0.0f;
        
        // Check if black would be in check with flipped turn
        std::string fen = board.to_fen();
        std::istringstream iss(fen);
        std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
        iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
        
        turn_str = "b";  // Flip to black
        std::string flipped_fen = board_str + " " + turn_str + " " + castling_str + " " + 
                                 ep_str + " " + halfmove_str + " " + fullmove_str;
        
        ChessBoard flipped_board(flipped_fen);
        if (!flipped_board.is_game_over()) {
            black_in_check = flipped_board.is_in_check(ChessBoard::BLACK) ? 1.0f : 0.0f;
        }
    } else {
        black_in_check = board.is_in_check(ChessBoard::BLACK) ? 1.0f : 0.0f;
        
        // Check if white would be in check with flipped turn
        std::string fen = board.to_fen();
        std::istringstream iss(fen);
        std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
        iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
        
        turn_str = "w";  // Flip to white
        std::string flipped_fen = board_str + " " + turn_str + " " + castling_str + " " + 
                                 ep_str + " " + halfmove_str + " " + fullmove_str;
        
        ChessBoard flipped_board(flipped_fen);
        if (!flipped_board.is_game_over()) {
            white_in_check = flipped_board.is_in_check(ChessBoard::WHITE) ? 1.0f : 0.0f;
        }
    }
    
    features[0] = white_in_check;
    features[1] = black_in_check;
    
    return features;
}

std::array<float, 2> FeatureExtractor::extract_mobility_features(const ChessBoard& board) {
    std::array<float, 2> mobility_features = {0.0f, 0.0f};  // [white_mobility, black_mobility]
    
    // Count pieces for each color
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str;
    iss >> board_str;
    
    int white_pieces = 0;
    int black_pieces = 0;
    
    for (char c : board_str) {
        if (c != '/' && !std::isdigit(c)) {
            if (std::isupper(c)) {
                white_pieces++;
            } else if (std::islower(c)) {
                black_pieces++;
            }
        }
    }
    
    // Only calculate white mobility if white has < 8 pieces
    if (white_pieces < 8) {
        float white_moves = 0.0f;
        
        if (board.turn() == ChessBoard::WHITE) {
            white_moves = static_cast<float>(board.get_legal_moves().size());
        } else {
            // Get white moves by flipping turn
            std::istringstream fen_stream(board.to_fen());
            std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
            fen_stream >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
            
            turn_str = "w";
            std::string flipped_fen = board_str + " " + turn_str + " " + castling_str + " " + 
                                     ep_str + " " + halfmove_str + " " + fullmove_str;
            
            ChessBoard flipped_board(flipped_fen);
            if (!flipped_board.is_game_over()) {
                white_moves = static_cast<float>(flipped_board.get_legal_moves().size());
            }
        }
        
        // Scale by white piece count: max((8 - white_pieces) / 6, 0)
        float white_factor = std::max((8.0f - white_pieces) / 6.0f, 0.0f);
        mobility_features[0] = white_factor * white_moves;
    }
    
    // Only calculate black mobility if black has < 8 pieces
    if (black_pieces < 8) {
        float black_moves = 0.0f;
        
        if (board.turn() == ChessBoard::BLACK) {
            black_moves = static_cast<float>(board.get_legal_moves().size());
        } else {
            // Get black moves by flipping turn
            std::istringstream fen_stream(board.to_fen());
            std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
            fen_stream >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
            
            turn_str = "b";
            std::string flipped_fen = board_str + " " + turn_str + " " + castling_str + " " + 
                                     ep_str + " " + halfmove_str + " " + fullmove_str;
            
            ChessBoard flipped_board(flipped_fen);
            if (!flipped_board.is_game_over()) {
                black_moves = static_cast<float>(flipped_board.get_legal_moves().size());
            }
        }
        
        // Scale by black piece count: max((8 - black_pieces) / 6, 0)
        float black_factor = std::max((8.0f - black_pieces) / 6.0f, 0.0f);
        mobility_features[1] = black_factor * black_moves;
    }
    
    return mobility_features;
}
