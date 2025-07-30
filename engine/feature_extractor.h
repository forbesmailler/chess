#pragma once
#include <vector>
#include <array>
#include <string>

class ChessBoard;

class FeatureExtractor {
public:
    static constexpr int FEATURE_SIZE = 1544; // 772 * 2 as in Python version
    
    static std::vector<float> extract_features(const std::string& fen);
    static std::vector<float> extract_features(const ChessBoard& board);
    
private:
    static std::array<float, 768> extract_piece_features(const ChessBoard& board);
    static std::array<float, 4> extract_castling_features(const ChessBoard& board);
};
