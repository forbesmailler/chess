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
    
    // Check current position's check status
    ChessBoard::Color current_turn = board.turn();
    bool current_in_check = board.is_in_check(current_turn);
    
    // Parse FEN once and reuse components
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Flip turn and check opponent
    std::string flipped_turn = (turn_str == "w") ? "b" : "w";
    std::string flipped_fen = board_str + " " + flipped_turn + " " + castling_str + " " + 
                             ep_str + " " + halfmove_str + " " + fullmove_str;
    
    bool opponent_in_check = false;
    ChessBoard flipped_board(flipped_fen);
    if (!flipped_board.is_game_over()) {
        ChessBoard::Color opponent_color = (current_turn == ChessBoard::WHITE) ? ChessBoard::BLACK : ChessBoard::WHITE;
        opponent_in_check = flipped_board.is_in_check(opponent_color);
    }
    
    // Set features based on colors (always white first, then black)
    if (current_turn == ChessBoard::WHITE) {
        features[0] = current_in_check ? 1.0f : 0.0f;   // white in check
        features[1] = opponent_in_check ? 1.0f : 0.0f;  // black in check
    } else {
        features[0] = opponent_in_check ? 1.0f : 0.0f;  // white in check
        features[1] = current_in_check ? 1.0f : 0.0f;   // black in check
    }
    
    return features;
}

std::array<float, 2> FeatureExtractor::extract_mobility_features(const ChessBoard& board) {
    std::array<float, 2> mobility_features = {0.0f, 0.0f};  // [white_mobility, black_mobility]
    
    // Parse FEN once to count pieces and extract components
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Count pieces for each color
    int white_pieces = 0;
    int black_pieces = 0;
    for (char c : board_str) {
        if (c != '/' && !std::isdigit(c)) {
            if (std::isupper(c)) white_pieces++;
            else if (std::islower(c)) black_pieces++;
        }
    }
    
    ChessBoard::Color current_turn = board.turn();
    auto current_moves = board.get_legal_moves();
    
    // Calculate white mobility if needed
    if (white_pieces < 8) {
        float white_moves = 0.0f;
        
        if (current_turn == ChessBoard::WHITE) {
            white_moves = static_cast<float>(current_moves.size());
        } else {
            // Get white moves by flipping turn
            std::string white_fen = board_str + " w " + castling_str + " " + 
                                   ep_str + " " + halfmove_str + " " + fullmove_str;
            ChessBoard white_board(white_fen);
            if (!white_board.is_game_over()) {
                white_moves = static_cast<float>(white_board.get_legal_moves().size());
            }
        }
        
        float white_factor = std::max((8.0f - white_pieces) / 6.0f, 0.0f);
        mobility_features[0] = white_factor * white_moves;
    }
    
    // Calculate black mobility if needed
    if (black_pieces < 8) {
        float black_moves = 0.0f;
        
        if (current_turn == ChessBoard::BLACK) {
            black_moves = static_cast<float>(current_moves.size());
        } else {
            // Get black moves by flipping turn
            std::string black_fen = board_str + " b " + castling_str + " " + 
                                   ep_str + " " + halfmove_str + " " + fullmove_str;
            ChessBoard black_board(black_fen);
            if (!black_board.is_game_over()) {
                black_moves = static_cast<float>(black_board.get_legal_moves().size());
            }
        }
        
        float black_factor = std::max((8.0f - black_pieces) / 6.0f, 0.0f);
        mobility_features[1] = black_factor * black_moves;
    }
    
    return mobility_features;
}
