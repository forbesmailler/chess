#include <gtest/gtest.h>

#include <cmath>

#include "chess_board.h"
#include "generated_config.h"
#include "handcrafted_eval.h"

TEST(HandcraftedEval, StartingPositionNearZero) {
    ChessBoard board;
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 500.0f);  // Sigmoid-scaled, near 0
}

TEST(HandcraftedEval, MaterialImbalanceDetected) {
    // Position with white missing a knight (black has extra knight)
    ChessBoard board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_LT(eval, -2000.0f);  // Sigmoid-scaled: missing knight is large disadvantage
}

TEST(HandcraftedEval, WhiteWinningMaterial) {
    // White has extra queen
    ChessBoard board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float base_eval = handcrafted_evaluate(board);

    // Remove black's queen
    ChessBoard board2("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    float eval2 = handcrafted_evaluate(board2);

    EXPECT_GT(eval2 - base_eval, 3000.0f);  // Sigmoid-scaled: queen advantage is large
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

    // Passed pawn should be valued higher (accounting for black's missing pawn
    // material) The passed pawn bonus should outweigh the material difference from
    // black having a pawn
    EXPECT_GT(passed_eval, blocked_eval);
}

TEST(HandcraftedEval, SymmetricPositionNearZero) {
    // A symmetric position should evaluate near zero
    ChessBoard board(
        "r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 4 3");
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 300.0f);  // Sigmoid-scaled, near 0
}

TEST(HandcraftedEval, EvalIsFinite) {
    ChessBoard board;
    float eval = handcrafted_evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
}
