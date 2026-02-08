#include <gtest/gtest.h>

#include <cmath>

#include "../chess_board.h"
#include "../handcrafted_eval.h"

TEST(HandcraftedEval, StartingPositionNearZero) {
    ChessBoard board;
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 50.0f);  // Within 50cp of 0
}

TEST(HandcraftedEval, MaterialImbalanceDetected) {
    // Position with white missing a knight (black has extra knight)
    ChessBoard board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_LT(eval, -200.0f);  // White should be clearly worse
}

TEST(HandcraftedEval, WhiteWinningMaterial) {
    // White has extra queen
    ChessBoard board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float base_eval = handcrafted_evaluate(board);

    // Remove black's queen
    ChessBoard board2("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float eval2 = handcrafted_evaluate(board2);

    EXPECT_GT(eval2 - base_eval, 700.0f);  // Queen worth ~900cp
}

TEST(HandcraftedEval, CheckmateWhiteLoses) {
    // Fool's mate - white is checkmated
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    float eval = handcrafted_evaluate(board);
    EXPECT_EQ(eval, -10000.0f);
}

TEST(HandcraftedEval, CheckmateBlackLoses) {
    // Black is checkmated
    // Scholar's mate position (after Qxf7#)
    ChessBoard board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");
    float eval = handcrafted_evaluate(board);
    EXPECT_EQ(eval, 10000.0f);  // Black is mated, positive from white's perspective
}

TEST(HandcraftedEval, StalemateIsZero) {
    // Black king in corner, white queen trapping but not checking
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    ASSERT_TRUE(board.is_stalemate());
    float eval = handcrafted_evaluate(board);
    EXPECT_EQ(eval, 0.0f);
}

TEST(HandcraftedEval, EndgameKingCentralisation) {
    // In endgame, centralised king should be better
    // King on e4 vs king on a1, with just pawns
    ChessBoard central("8/8/8/8/4K3/8/PPPP4/8 w - - 0 1");
    ChessBoard corner("K7/8/8/8/8/8/PPPP4/8 w - - 0 1");

    float central_eval = handcrafted_evaluate(central);
    float corner_eval = handcrafted_evaluate(corner);

    EXPECT_GT(central_eval, corner_eval);
}

TEST(HandcraftedEval, PassedPawnBonus) {
    // White has a passed pawn on e5
    ChessBoard with_passed("8/8/8/4P3/8/8/8/4K2k w - - 0 1");
    // White has a pawn blocked by enemy pawn
    ChessBoard blocked("8/4p3/8/4P3/8/8/8/4K2k w - - 0 1");

    float passed_eval = handcrafted_evaluate(with_passed);
    float blocked_eval = handcrafted_evaluate(blocked);

    // Passed pawn should be valued higher (accounting for black's missing pawn material)
    // The passed pawn bonus should outweigh the material difference from black having a pawn
    EXPECT_GT(passed_eval, blocked_eval);
}

TEST(HandcraftedEval, SymmetricPositionNearZero) {
    // A symmetric position should evaluate near zero
    ChessBoard board("r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 4 3");
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 30.0f);
}

TEST(HandcraftedEval, EvalIsFinite) {
    ChessBoard board;
    float eval = handcrafted_evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
}
