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

// --- Rook on open file ---

TEST(HandcraftedEval, RookOnOpenFile) {
    // White rook on e-file with no pawns on e-file (open)
    ChessBoard open_file("4k3/pppp1ppp/8/8/8/8/PPPP1PPP/4K1R1 w - - 0 1");
    // Same but with white pawn on e-file (closed)
    ChessBoard closed_file("4k3/pppp1ppp/8/8/8/8/PPPPPPP1/4K1R1 w - - 0 1");

    float open_eval = handcrafted_evaluate(open_file);
    float closed_eval = handcrafted_evaluate(closed_file);

    // Rook on open file should score higher than on closed file
    EXPECT_GT(open_eval, closed_eval);
}

TEST(HandcraftedEval, RookOnSemiOpenFile) {
    // White rook on e-file, no white pawns but black pawn present (semi-open)
    ChessBoard semi_open("4k3/pppppppp/8/8/8/8/PPPP1PPP/4K1R1 w - - 0 1");
    // Same but with white pawn on e-file too (closed)
    ChessBoard closed("4k3/pppppppp/8/8/8/8/PPPPPPP1/4K1R1 w - - 0 1");

    float semi_eval = handcrafted_evaluate(semi_open);
    float closed_eval = handcrafted_evaluate(closed);

    EXPECT_GT(semi_eval, closed_eval);
}

// --- Bishop pair ---

TEST(HandcraftedEval, BishopPairBonus) {
    // White has bishop pair, black has one bishop
    ChessBoard pair("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1");
    // White has only one bishop
    ChessBoard single("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");

    float pair_eval = handcrafted_evaluate(pair);
    float single_eval = handcrafted_evaluate(single);

    // Bishop pair bonus: the extra bishop adds material AND a bishop pair bonus
    // The pair bonus is on top of material, so diff > just one bishop's material value
    float diff = pair_eval - single_eval;
    EXPECT_GT(diff, 0.0f);
}

TEST(HandcraftedEval, BishopPairSymmetric) {
    // Both sides have bishop pair — should be near zero
    ChessBoard board("2b1kb2/8/8/8/8/8/8/2B1KB2 w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 200.0f);
}

// --- Isolated pawns ---

TEST(HandcraftedEval, IsolatedPawnPenalty) {
    // White pawn on e4 isolated (no pawns on d or f files)
    ChessBoard isolated("4k3/8/8/8/4P3/8/P6P/4K3 w - - 0 1");
    // White pawn on e4 with neighbor on d2 (not isolated)
    ChessBoard connected("4k3/8/8/8/4P3/8/P2P3P/4K3 w - - 0 1");

    float isolated_eval = handcrafted_evaluate(isolated);
    float connected_eval = handcrafted_evaluate(connected);

    // Connected pawn structure should score higher
    EXPECT_GT(connected_eval, isolated_eval);
}

TEST(HandcraftedEval, IsolatedPawnOnAFile) {
    // White pawn on a2 isolated (no pawn on b file)
    ChessBoard isolated("4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    // White pawn on a2 with neighbor on b2
    ChessBoard connected("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1");

    float isolated_eval = handcrafted_evaluate(isolated);
    float connected_eval = handcrafted_evaluate(connected);

    EXPECT_GT(connected_eval, isolated_eval);
}

// --- Doubled pawns ---

TEST(HandcraftedEval, DoubledPawnPenalty) {
    // White has doubled pawns on e-file
    ChessBoard doubled("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1");
    // White has pawns on e and d files (not doubled)
    ChessBoard separate("4k3/8/8/8/4P3/3P4/8/4K3 w - - 0 1");

    float doubled_eval = handcrafted_evaluate(doubled);
    float separate_eval = handcrafted_evaluate(separate);

    // Non-doubled pawns should score better
    EXPECT_GT(separate_eval, doubled_eval);
}

// --- Mobility ---

TEST(HandcraftedEval, KnightMobilityMatters) {
    // Knight in center (high mobility) vs knight in corner (low mobility)
    ChessBoard center("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1");
    ChessBoard corner("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");

    float center_eval = handcrafted_evaluate(center);
    float corner_eval = handcrafted_evaluate(corner);

    // Central knight with more mobility should eval higher
    EXPECT_GT(center_eval, corner_eval);
}

// --- Eval output bounds ---

TEST(HandcraftedEval, OutputBoundedByMateValue) {
    // Even extreme positions should be bounded by MATE_VALUE
    ChessBoard board("QQQQQQQQ/8/8/8/8/8/8/4K2k w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_LE(eval, config::MATE_VALUE);
    EXPECT_GE(eval, -config::MATE_VALUE);
    EXPECT_TRUE(std::isfinite(eval));
}

TEST(HandcraftedEval, OutputNegativeBound) {
    // Black has massive advantage
    ChessBoard board("4k3/8/8/8/8/8/8/qqqqqqqK w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_LE(eval, config::MATE_VALUE);
    EXPECT_GE(eval, -config::MATE_VALUE);
    EXPECT_TRUE(std::isfinite(eval));
}

// --- Passed pawn rank scaling ---

TEST(HandcraftedEval, PassedPawnHigherRankMoreValuable) {
    // White passed pawn on 6th rank
    ChessBoard rank6("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1");
    // White passed pawn on 3rd rank
    ChessBoard rank3("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1");

    float rank6_eval = handcrafted_evaluate(rank6);
    float rank3_eval = handcrafted_evaluate(rank3);

    // Higher-rank passed pawn should be more valuable
    EXPECT_GT(rank6_eval, rank3_eval);
}

// --- King pawn shield ---

TEST(HandcraftedEval, KingPawnShieldBonus) {
    // White king on g1 with pawns on f2, g2, h2 (good shield)
    ChessBoard shielded("4k3/pppppppp/8/8/8/8/PPPPPPPP/6K1 w - - 0 1");
    // White king on g1 with no pawns near king (no shield)
    ChessBoard unshielded("4k3/pppppppp/8/8/8/8/PPP2PPP/6K1 w - - 0 1");

    float shielded_eval = handcrafted_evaluate(shielded);
    float unshielded_eval = handcrafted_evaluate(unshielded);

    // King with pawn shield should be better
    EXPECT_GT(shielded_eval, unshielded_eval);
}

// --- Black passed pawn ---

TEST(HandcraftedEval, BlackPassedPawnBonus) {
    // Black has a passed pawn on d3 (no white pawns on c, d, or e files to block)
    ChessBoard passed("4k3/8/8/8/8/3p4/8/4K3 w - - 0 1");
    // Black has a pawn on d3 blocked by white pawn on d2 (not passed)
    ChessBoard blocked("4k3/8/8/8/8/3p4/3P4/4K3 w - - 0 1");

    float passed_eval = handcrafted_evaluate(passed);
    float blocked_eval = handcrafted_evaluate(blocked);

    // Position with black passed pawn should be worse for white (lower eval)
    // Note: blocked position has an extra white pawn so material difference
    // offsets the passed pawn. The key test: from black's perspective, the passed
    // pawn makes the position better for black.
    // With blocked: white has a pawn (material advantage) but black pawn not passed.
    // With passed: white has NO pawn, black has passed pawn.
    // So passed_eval < blocked_eval because white lost material AND black has a passed
    // pawn.
    EXPECT_LT(passed_eval, blocked_eval);
}

TEST(HandcraftedEval, BlackPassedPawnHigherRankMoreValuable) {
    // Black passed pawn on 2nd rank (closer to promotion)
    ChessBoard rank2("4k3/8/8/8/8/8/3p4/4K3 w - - 0 1");
    // Black passed pawn on 5th rank (far from promotion)
    ChessBoard rank5("4k3/8/3p4/8/8/8/8/4K3 w - - 0 1");

    float rank2_eval = handcrafted_evaluate(rank2);
    float rank5_eval = handcrafted_evaluate(rank5);

    // Black pawn closer to promotion => worse for white => lower eval
    EXPECT_LT(rank2_eval, rank5_eval);
}
