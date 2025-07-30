#include "feature_extractor.h"
#include "chess_board.h"
#include <algorithm>
#include <sstream>
#include <cctype>

std::vector<float> FeatureExtractor::extract_features(const std::string& fen) {
    ChessBoard board(fen);
    return extract_features(board);
}

std::vector<float> FeatureExtractor::extract_features(const ChessBoard& board) {
    // Extract piece features (12 piece types * 64 squares = 768)
    auto piece_features = extract_piece_features(board);
    
    // Extract castling features (4)
    auto castling_features = extract_castling_features(board);
    
    // Combine base features
    std::vector<float> base_features;
    base_features.reserve(772);
    
    // Add piece features
    for (float f : piece_features) {
        base_features.push_back(f);
    }
    
    // Add castling features
    for (float f : castling_features) {
        base_features.push_back(f);
    }
    
    // Calculate piece count factor (matching Python logic)
    int n_pieces = board.piece_count();
    
    float factor = static_cast<float>(n_pieces - 2) / 30.0f;
    
    // Create final feature vector (772 * 2 = 1544)
    std::vector<float> features;
    features.reserve(FEATURE_SIZE);
    
    // First half: base * factor
    for (float f : base_features) {
        features.push_back(f * factor);
    }
    
    // Second half: base * (1 - factor)
    for (float f : base_features) {
        features.push_back(f * (1.0f - factor));
    }
    
    return features;
}

std::array<float, 768> FeatureExtractor::extract_piece_features(const ChessBoard& board) {
    std::array<float, 768> piece_arr;
    piece_arr.fill(0.0f);
    
    // Parse the FEN to get piece positions
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str;
    iss >> board_str;
    
    int square = 56; // Start at a8 (rank 8, file a)
    
    for (char c : board_str) {
        if (c == '/') {
            square -= 16; // Move to next rank down
        } else if (std::isdigit(c)) {
            square += (c - '0'); // Skip empty squares
        } else {
            // Map piece character to our piece index
            int piece_type = 0;
            bool is_white = std::isupper(c);
            
            switch (std::tolower(c)) {
                case 'p': piece_type = 1; break; // Pawn
                case 'n': piece_type = 2; break; // Knight
                case 'b': piece_type = 3; break; // Bishop
                case 'r': piece_type = 4; break; // Rook
                case 'q': piece_type = 5; break; // Queen
                case 'k': piece_type = 6; break; // King
            }
            
            if (piece_type > 0) {
                // Calculate index: piece_type - 1 for white (0-5), + 6 for black (6-11)
                int idx = (piece_type - 1) + (is_white ? 0 : 6);
                piece_arr[idx * 64 + square] = 1.0f;
            }
            
            square++;
        }
    }
    
    return piece_arr;
}

std::array<float, 4> FeatureExtractor::extract_castling_features(const ChessBoard& board) {
    auto castling = board.get_castling_rights();
    return {
        castling.white_kingside ? 1.0f : 0.0f,
        castling.white_queenside ? 1.0f : 0.0f,
        castling.black_kingside ? 1.0f : 0.0f,
        castling.black_queenside ? 1.0f : 0.0f
    };
}
