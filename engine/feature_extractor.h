#pragma once
#include <array>
#include <string>
#include <vector>

class ChessBoard;

class FeatureExtractor {
   public:
    static constexpr int FEATURE_SIZE =
        1542;  // 770 * 2 + 2 (768 piece + 2 check features, scaled twice, plus 2 mobility)

    static std::vector<float> extract_features(const std::string& fen);
    static std::vector<float> extract_features(const ChessBoard& board);

   private:
    static std::array<float, 768> extract_piece_features(const ChessBoard& board);
    static std::array<float, 2> extract_additional_features(const ChessBoard& board);
    static std::array<float, 2> extract_mobility_features(const ChessBoard& board);
};
