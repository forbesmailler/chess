#pragma once
#include <vector>
#include <array>
#include <string>

class ChessBoard;

class FeatureExtractor {
public:
    static constexpr int FEATURE_SIZE = 1552; // 776 * 2 (768 piece + 8 additional features)
    
    static std::vector<float> extract_features(const std::string& fen);
    static std::vector<float> extract_features(const ChessBoard& board);
    
private:
    static std::array<float, 768> extract_piece_features(const ChessBoard& board);
    static std::array<float, 8> extract_additional_features(const ChessBoard& board);
};
