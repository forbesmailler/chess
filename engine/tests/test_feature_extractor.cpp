#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "../chess_board.h"
#include "../feature_extractor.h"

TEST(FeatureExtractorTest, FeatureSize) {
    ChessBoard board;
    auto features = FeatureExtractor::extract_features(board);
    EXPECT_EQ(features.size(), FeatureExtractor::FEATURE_SIZE);
    EXPECT_EQ(features.size(), 1542);
}

TEST(FeatureExtractorTest, FeatureSizeFromFen) {
    auto features = FeatureExtractor::extract_features(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    EXPECT_EQ(features.size(), 1542);
}

TEST(FeatureExtractorTest, DifferentPositionsDifferentFeatures) {
    ChessBoard board1;
    ChessBoard board2("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

    auto features1 = FeatureExtractor::extract_features(board1);
    auto features2 = FeatureExtractor::extract_features(board2);

    EXPECT_NE(features1, features2);
}

TEST(FeatureExtractorTest, SamePositionSameFeatures) {
    ChessBoard board1;
    ChessBoard board2;

    auto features1 = FeatureExtractor::extract_features(board1);
    auto features2 = FeatureExtractor::extract_features(board2);

    EXPECT_EQ(features1, features2);
}

TEST(FeatureExtractorTest, FeaturesAreFinite) {
    ChessBoard board;
    auto features = FeatureExtractor::extract_features(board);

    for (size_t i = 0; i < features.size(); ++i) {
        EXPECT_TRUE(std::isfinite(features[i])) << "Feature " << i << " is not finite";
    }
}

TEST(FeatureExtractorTest, EndgamePosition) {
    // King and rook endgame - should have mobility features active
    ChessBoard board("4k3/8/8/8/8/8/8/4K2R w - - 0 1");
    auto features = FeatureExtractor::extract_features(board);
    EXPECT_EQ(features.size(), 1542);

    // Features should still be finite
    for (size_t i = 0; i < features.size(); ++i) {
        EXPECT_TRUE(std::isfinite(features[i])) << "Feature " << i << " is not finite";
    }
}

TEST(FeatureExtractorTest, CheckPosition) {
    // White in check
    ChessBoard board("rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2");
    auto features = FeatureExtractor::extract_features(board);
    EXPECT_EQ(features.size(), 1542);
}
