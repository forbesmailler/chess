#include <gtest/gtest.h>

#include <memory>

#include "chess_board.h"
#include "chess_engine.h"
#include "mcts_engine.h"

class ChessEngineTest : public ::testing::Test {};

TEST_F(ChessEngineTest, NegamaxFindsMove) {
    ChessEngine engine(100);  // 100ms max time
    ChessBoard board;

    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);

    EXPECT_FALSE(result.best_move.uci().empty());
    EXPECT_GT(result.nodes_searched, 0);
}

TEST_F(ChessEngineTest, NegamaxFindsMateInOne) {
    ChessEngine engine(1000);
    // White to move, Qh7# is mate in one
    ChessBoard board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");

    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);

    EXPECT_EQ(result.best_move.uci(), "h5f7");  // Qxf7# (scholar's mate)
}

TEST_F(ChessEngineTest, EvaluateReturnsFinite) {
    ChessEngine engine(100);
    ChessBoard board;

    float eval = engine.evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
}

TEST_F(ChessEngineTest, EvaluateCheckmate) {
    ChessEngine engine(100);
    // Fool's mate - white is checkmated
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    float eval = engine.evaluate(board);
    // White is mated, so from white's perspective this should be very negative
    EXPECT_LT(eval, -100.0f);
}

TEST_F(ChessEngineTest, MCTSFindsMove) {
    MCTSEngine engine(100);  // 100ms max time
    ChessBoard board;

    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);

    EXPECT_FALSE(result.best_move.uci().empty());
    EXPECT_GT(result.nodes_searched, 0);
}

TEST_F(ChessEngineTest, MCTSEvaluateReturnsFinite) {
    MCTSEngine engine(100);
    ChessBoard board;

    float eval = engine.evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
}

TEST_F(ChessEngineTest, TimeControlRespected) {
    ChessEngine engine(500);  // 500ms max
    ChessBoard board;

    auto start = std::chrono::steady_clock::now();
    TimeControl tc{1000, 0, 0};  // 1 second on clock
    auto result = engine.get_best_move(board, tc);
    auto elapsed = std::chrono::steady_clock::now() - start;

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    // Should complete within reasonable time (allow some overhead)
    EXPECT_LT(elapsed_ms, 2000);
}

TEST_F(ChessEngineTest, SearchResultHasValidDepth) {
    ChessEngine engine(200);
    ChessBoard board;

    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);

    EXPECT_GE(result.depth, 1);
}
