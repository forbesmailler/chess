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
    auto castling_features = extract_castling_features(board);
    
    std::vector<float> base_features;
    base_features.reserve(772);
    
    for (float f : piece_features) base_features.push_back(f);
    for (float f : castling_features) base_features.push_back(f);
    
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

std::array<float, 4> FeatureExtractor::extract_castling_features(const ChessBoard& board) {
    auto castling = board.get_castling_rights();
    return {
        castling.white_kingside ? 1.0f : 0.0f,
        castling.white_queenside ? 1.0f : 0.0f,
        castling.black_kingside ? 1.0f : 0.0f,
        castling.black_queenside ? 1.0f : 0.0f
    };
}
