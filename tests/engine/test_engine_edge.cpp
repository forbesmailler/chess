#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

#include "chess_board.h"
#include "chess_engine.h"
#include "generated_config.h"
#include "handcrafted_eval.h"
#include "mcts_engine.h"
#include "nnue_model.h"
#include "self_play.h"

namespace {
std::string create_nnue_weights_buffer(unsigned int seed = 42) {
    constexpr int INPUT = config::nnue::INPUT_SIZE;
    constexpr int H1 = config::nnue::HIDDEN1_SIZE;
    constexpr int H2 = config::nnue::HIDDEN2_SIZE;
    constexpr int OUTPUT = config::nnue::OUTPUT_SIZE;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    std::ostringstream oss;
    oss.write("NNUE", 4);
    uint32_t header[] = {1, INPUT, H1, H2, OUTPUT};
    oss.write(reinterpret_cast<char*>(header), sizeof(header));

    auto write_random = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            float val = dist(rng);
            oss.write(reinterpret_cast<char*>(&val), sizeof(float));
        }
    };

    write_random(INPUT * H1);
    write_random(H1);
    write_random(H1 * H2);
    write_random(H2);
    write_random(H2 * OUTPUT);
    write_random(OUTPUT);

    return oss.str();
}

std::shared_ptr<NNUEModel> load_nnue_from_buffer(unsigned int seed = 42) {
    auto model = std::make_shared<NNUEModel>();
    std::string buf = create_nnue_weights_buffer(seed);
    std::istringstream stream(buf);
    model->load_weights(stream);
    return model;
}
// Expose protected calculate_search_time for direct testing
class TestableEngine : public ChessEngine {
   public:
    using ChessEngine::ChessEngine;
    int test_calculate_time(const TimeControl& tc) { return calculate_search_time(tc); }
};
}  // namespace

// --- BaseEngine: calculate_search_time exact values ---

TEST(BaseEngineEdge, CalculateSearchTimeZeroReturnsMax) {
    TestableEngine engine(200);
    TimeControl tc{0, 0, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 200);
}

TEST(BaseEngineEdge, CalculateSearchTimeNegativeReturnsMax) {
    TestableEngine engine(200);
    TimeControl tc{-100, 0, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 200);
}

TEST(BaseEngineEdge, CalculateSearchTimeCappedByMax) {
    TestableEngine engine(500);
    // allocated = 1000 + 4000/40 = 1100, capped at 500
    TimeControl tc{4000, 1000, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 500);
}

TEST(BaseEngineEdge, CalculateSearchTimeUncapped) {
    TestableEngine engine(5000);
    // allocated = 1000 + 4000/40 = 1100, uncapped (< 5000)
    TimeControl tc{4000, 1000, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 1100);
}

TEST(BaseEngineEdge, CalculateSearchTimeNoIncrement) {
    TestableEngine engine(5000);
    // allocated = 0 + 40000/40 = 1000
    TimeControl tc{40000, 0, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 1000);
}

TEST(BaseEngineEdge, CalculateSearchTimeOnlyIncrement) {
    TestableEngine engine(5000);
    // allocated = 3000 + 1/40 = 3000 (integer division 1/40 = 0)
    TimeControl tc{1, 3000, 0};
    EXPECT_EQ(engine.test_calculate_time(tc), 3000);
}

TEST(BaseEngineEdge, SetGetMaxTime) {
    ChessEngine engine(500);
    EXPECT_EQ(engine.get_max_time(), 500);

    engine.set_max_time(1000);
    EXPECT_EQ(engine.get_max_time(), 1000);

    engine.set_max_time(0);
    EXPECT_EQ(engine.get_max_time(), 0);
}

TEST(BaseEngineEdge, EvalModeDefault) {
    ChessEngine engine;
    EXPECT_EQ(engine.get_eval_mode(), EvalMode::HANDCRAFTED);
}

TEST(BaseEngineEdge, EvalModeNNUE) {
    ChessEngine engine(1000, EvalMode::NNUE, nullptr);
    EXPECT_EQ(engine.get_eval_mode(), EvalMode::NNUE);
}

TEST(BaseEngineEdge, StopSearchPreventsLongSearch) {
    ChessEngine engine(5000);
    ChessBoard board;

    // Stop immediately — the search should still produce a result
    engine.stop_search();
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    // Engine should still return a valid move even when stopped
    EXPECT_FALSE(result.best_move.uci().empty());
}

// --- BaseEngine: raw_evaluate via ChessEngine::evaluate ---

TEST(BaseEngineEdge, EvaluateStalemateReturnsZero) {
    ChessEngine engine(100);
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    float eval = engine.evaluate(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

TEST(BaseEngineEdge, EvaluateNNUEFallbackToHandcrafted) {
    // NNUE mode with nullptr model should fall back to handcrafted
    ChessEngine engine(100, EvalMode::NNUE, nullptr);
    ChessBoard board;
    float eval = engine.evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
}

// --- BaseEngine: time allocation with zero/negative time left ---

TEST(BaseEngineEdge, SearchWithZeroTimeLeft) {
    ChessEngine engine(100);
    ChessBoard board;
    TimeControl tc{0, 0, 0};  // zero time left
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
}

TEST(BaseEngineEdge, SearchWithNegativeTimeLeft) {
    ChessEngine engine(100);
    ChessBoard board;
    TimeControl tc{-1000, 0, 0};  // negative time
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
}

TEST(BaseEngineEdge, SearchWithIncrement) {
    ChessEngine engine(500);
    ChessBoard board;
    TimeControl tc{10000, 5000, 0};  // 10s + 5s increment
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
    EXPECT_GE(result.depth, 1);
}

// --- ChessEngine: clear_caches ---

TEST(ChessEngineEdge, ClearCachesDoesNotCrash) {
    ChessEngine engine(100);
    ChessBoard board;

    // Run a search to populate caches
    TimeControl tc{60000, 0, 0};
    engine.get_best_move(board, tc);

    // Clear caches and search again
    engine.clear_caches();
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
    EXPECT_GE(result.depth, 1);
}

TEST(ChessEngineEdge, ClearCachesBeforeSearch) {
    ChessEngine engine(100);
    engine.clear_caches();

    ChessBoard board;
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
}

// --- Handcrafted eval: specific piece arrangements ---

TEST(HandcraftedEvalEdge, KingVsKing) {
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 100.0f);  // Should be approximately equal
}

TEST(HandcraftedEvalEdge, OnlyPawns) {
    ChessBoard board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_NEAR(eval, 0.0f, 300.0f);  // Symmetric, near zero
}

TEST(HandcraftedEvalEdge, MassiveWhiteAdvantage) {
    // White has queen + rook vs bare king
    ChessBoard board("4k3/8/8/8/8/8/8/QR2K3 w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_GT(eval, 5000.0f);
}

TEST(HandcraftedEvalEdge, MassiveBlackAdvantage) {
    // Black has queen + rook vs bare king
    ChessBoard board("qr2k3/8/8/8/8/8/8/4K3 w - - 0 1");
    float eval = handcrafted_evaluate(board);
    EXPECT_LT(eval, -5000.0f);
}

// --- SelfPlay: piece encoding tested indirectly via encode_position ---

TEST(SelfPlayEdge, EncodedPiecesNonZeroForOccupiedSquares) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);

    // Check that rank 1 (squares 0-7) have non-zero nibbles (white pieces)
    for (int sq = 0; sq < 8; sq += 2) {
        uint8_t byte = pos.piece_placement[sq / 2];
        uint8_t high = (byte >> 4) & 0x0F;
        uint8_t low = byte & 0x0F;
        EXPECT_GT(high, 0) << "Square " << sq << " should be occupied";
        EXPECT_GT(low, 0) << "Square " << (sq + 1) << " should be occupied";
    }

    // Check that rank 2 (squares 8-15) have non-zero nibbles (white pawns)
    for (int sq = 8; sq < 16; sq += 2) {
        uint8_t byte = pos.piece_placement[sq / 2];
        uint8_t high = (byte >> 4) & 0x0F;
        uint8_t low = byte & 0x0F;
        EXPECT_GT(high, 0) << "Square " << sq << " should be a pawn";
        EXPECT_GT(low, 0) << "Square " << (sq + 1) << " should be a pawn";
    }

    // Check that empty squares (ranks 3-6, squares 16-47) have zero nibbles
    for (int sq = 16; sq < 48; sq += 2) {
        uint8_t byte = pos.piece_placement[sq / 2];
        EXPECT_EQ(byte, 0) << "Byte " << sq / 2 << " should be empty";
    }
}

TEST(SelfPlayEdge, EncodedPieceKingVsKing) {
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);

    // Only two non-zero nibbles in the entire placement
    int non_zero_count = 0;
    for (int sq = 0; sq < 64; ++sq) {
        uint8_t byte = pos.piece_placement[sq / 2];
        uint8_t nibble = (sq % 2 == 0) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        if (nibble != 0) non_zero_count++;
    }
    EXPECT_EQ(non_zero_count, 2);
}

// --- SelfPlay: encode_piece exact nibble values ---

TEST(SelfPlayEdge, EncodePositionExactPieceNibbles) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);

    auto nibble_at = [&](int sq) -> uint8_t {
        uint8_t byte = pos.piece_placement[sq / 2];
        return (sq % 2 == 0) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    };

    // Encoding: 1=wP 2=wN 3=wB 4=wR 5=wQ 6=wK 7=bP 8=bN 9=bB 10=bR 11=bQ 12=bK

    // White rank 1: R(0)=4 N(1)=2 B(2)=3 Q(3)=5 K(4)=6 B(5)=3 N(6)=2 R(7)=4
    EXPECT_EQ(nibble_at(0), 4);  // a1 white rook
    EXPECT_EQ(nibble_at(1), 2);  // b1 white knight
    EXPECT_EQ(nibble_at(2), 3);  // c1 white bishop
    EXPECT_EQ(nibble_at(3), 5);  // d1 white queen
    EXPECT_EQ(nibble_at(4), 6);  // e1 white king
    EXPECT_EQ(nibble_at(5), 3);  // f1 white bishop
    EXPECT_EQ(nibble_at(6), 2);  // g1 white knight
    EXPECT_EQ(nibble_at(7), 4);  // h1 white rook

    // White rank 2: all white pawns = 1
    for (int sq = 8; sq < 16; ++sq) {
        EXPECT_EQ(nibble_at(sq), 1) << "sq " << sq << " should be white pawn (1)";
    }

    // Black rank 7: all black pawns = 7
    for (int sq = 48; sq < 56; ++sq) {
        EXPECT_EQ(nibble_at(sq), 7) << "sq " << sq << " should be black pawn (7)";
    }

    // Black rank 8: r(56)=10 n(57)=8 b(58)=9 q(59)=11 k(60)=12 b(61)=9 n(62)=8 r(63)=10
    EXPECT_EQ(nibble_at(56), 10);  // a8 black rook
    EXPECT_EQ(nibble_at(57), 8);   // b8 black knight
    EXPECT_EQ(nibble_at(58), 9);   // c8 black bishop
    EXPECT_EQ(nibble_at(59), 11);  // d8 black queen
    EXPECT_EQ(nibble_at(60), 12);  // e8 black king
    EXPECT_EQ(nibble_at(61), 9);   // f8 black bishop
    EXPECT_EQ(nibble_at(62), 8);   // g8 black knight
    EXPECT_EQ(nibble_at(63), 10);  // h8 black rook

    // Empty squares
    for (int sq = 16; sq < 48; ++sq) {
        EXPECT_EQ(nibble_at(sq), 0) << "sq " << sq << " should be empty (0)";
    }
}

// --- SelfPlay: partial castling rights encoding ---

TEST(SelfPlayEdge, EncodePositionPartialCastling) {
    // Only white kingside and black queenside
    ChessBoard board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    // WK=1, WQ=0, BK=0, BQ=1 -> (1<<3) | (0<<2) | (0<<1) | 1 = 9
    EXPECT_EQ(pos.castling, 9);
}

TEST(SelfPlayEdge, EncodePositionOnlyBlackCastling) {
    ChessBoard board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w kq - 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    // WK=0, WQ=0, BK=1, BQ=1 -> (0<<3) | (0<<2) | (1<<1) | 1 = 3
    EXPECT_EQ(pos.castling, 3);
}

TEST(SelfPlayEdge, EncodePositionAllCastling) {
    ChessBoard board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    // All 4 bits set = 15
    EXPECT_EQ(pos.castling, 15);
}

// --- SelfPlay: encode_position edge cases ---

TEST(SelfPlayEdge, EncodePositionBlackToMove) {
    ChessBoard board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, -50.0f, 0, 1);
    EXPECT_EQ(pos.side_to_move, 1);
    EXPECT_FLOAT_EQ(pos.search_eval, -50.0f);
    EXPECT_EQ(pos.game_result, 0);
    EXPECT_EQ(pos.ply_number, 1);
}

TEST(SelfPlayEdge, EncodePositionEnPassant) {
    // Black pawn on d4 can capture white pawn on e4 en passant (ep square e3)
    ChessBoard board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    EXPECT_EQ(pos.en_passant_file, 4);  // e-file
}

TEST(SelfPlayEdge, EncodePositionNoEnPassant) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    EXPECT_EQ(pos.en_passant_file, 255);
}

TEST(SelfPlayEdge, EncodePositionMaxPly) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 65535);
    EXPECT_EQ(pos.ply_number, 65535);
}

TEST(SelfPlayEdge, EncodePositionZeroPly) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    EXPECT_EQ(pos.ply_number, 0);
}

TEST(SelfPlayEdge, EncodePositionNegativeEval) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, -10000.0f, 0, 100);
    EXPECT_FLOAT_EQ(pos.search_eval, -10000.0f);
    EXPECT_EQ(pos.game_result, 0);
}

TEST(SelfPlayEdge, EncodePositionNoCastling) {
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
    EXPECT_EQ(pos.castling, 0);
}

// --- ModelComparator::Result::improved ---

TEST(ModelComparatorEdge, ImprovedWhenNewWinsMore) {
    ModelComparator::Result result;
    result.new_wins = 10;
    result.old_wins = 5;
    result.draws = 3;
    EXPECT_TRUE(result.improved());
}

TEST(ModelComparatorEdge, NotImprovedWhenOldWinsMore) {
    ModelComparator::Result result;
    result.new_wins = 3;
    result.old_wins = 10;
    result.draws = 5;
    EXPECT_FALSE(result.improved());
}

TEST(ModelComparatorEdge, NotImprovedWhenTied) {
    ModelComparator::Result result;
    result.new_wins = 5;
    result.old_wins = 5;
    result.draws = 10;
    EXPECT_FALSE(result.improved());
}

TEST(ModelComparatorEdge, NotImprovedWhenAllDraws) {
    ModelComparator::Result result;
    result.new_wins = 0;
    result.old_wins = 0;
    result.draws = 20;
    EXPECT_FALSE(result.improved());
}

TEST(ModelComparatorEdge, NotImprovedWhenAllZero) {
    ModelComparator::Result result;
    EXPECT_FALSE(result.improved());
}

// --- read_position from empty/short file ---

TEST(SelfPlayEdge, ReadPositionFromEmptyFile) {
    std::string tmp_file = "test_empty_read.bin";
    {
        std::ofstream out(tmp_file, std::ios::binary);
        // Write nothing
    }

    TrainingPosition pos;
    std::ifstream in(tmp_file, std::ios::binary);
    EXPECT_FALSE(SelfPlayGenerator::read_position(in, pos));

    std::remove(tmp_file.c_str());
}

TEST(SelfPlayEdge, ReadPositionFromTruncatedFile) {
    std::string tmp_file = "test_truncated_read.bin";
    {
        std::ofstream out(tmp_file, std::ios::binary);
        // Write only 10 bytes (less than 42)
        char data[10] = {};
        out.write(data, 10);
    }

    TrainingPosition pos;
    std::ifstream in(tmp_file, std::ios::binary);
    EXPECT_FALSE(SelfPlayGenerator::read_position(in, pos));

    std::remove(tmp_file.c_str());
}

// --- NNUE model error handling ---

TEST(NNUEModelEdge, LoadTruncatedStream) {
    std::ostringstream oss;
    oss.write("NNUE", 4);
    uint32_t header[] = {1, config::nnue::INPUT_SIZE, config::nnue::HIDDEN1_SIZE,
                         config::nnue::HIDDEN2_SIZE, config::nnue::OUTPUT_SIZE};
    oss.write(reinterpret_cast<char*>(header), sizeof(header));
    float val = 1.0f;
    oss.write(reinterpret_cast<char*>(&val), sizeof(float));

    std::istringstream stream(oss.str());
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(stream));
    EXPECT_FALSE(model.is_loaded());
}

TEST(NNUEModelEdge, LoadArchitectureMismatch) {
    std::ostringstream oss;
    oss.write("NNUE", 4);
    uint32_t header[] = {1, 100, 50, 10, 1};
    oss.write(reinterpret_cast<char*>(header), sizeof(header));

    std::istringstream stream(oss.str());
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(stream));
    EXPECT_FALSE(model.is_loaded());
}

TEST(NNUEModelEdge, LoadEmptyStream) {
    std::istringstream stream("");
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(stream));
    EXPECT_FALSE(model.is_loaded());
}

TEST(NNUEModelEdge, PredictUnloadedMultiplePositions) {
    NNUEModel model;
    ChessBoard board1;
    ChessBoard board2("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    EXPECT_FLOAT_EQ(model.predict(board1), 0.0f);
    EXPECT_FLOAT_EQ(model.predict(board2), 0.0f);
}

// --- ChessEngine: get_best_move edge cases ---

TEST(ChessEngineEdge, GetBestMoveCheckmate) {
    // White is checkmated — no legal moves, in check
    ChessEngine engine(100);
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_TRUE(result.best_move.uci().empty());
    EXPECT_FLOAT_EQ(result.score, -config::MATE_VALUE);
    EXPECT_EQ(result.depth, 0);
    EXPECT_EQ(result.nodes_searched, 0);
}

TEST(ChessEngineEdge, GetBestMoveStalemate) {
    // Black is stalemated — no legal moves, NOT in check
    ChessEngine engine(100);
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_TRUE(result.best_move.uci().empty());
    EXPECT_FLOAT_EQ(result.score, 0.0f);
    EXPECT_EQ(result.depth, 0);
}

TEST(ChessEngineEdge, GetBestMoveSingleLegalMove) {
    // Position with only one legal move
    // King on h8 in check by Qf7, only legal move is Kh8->Kg8 (if that's the only one)
    // Use: White Kf6, Qf7, Pawn h7, Black Kh8 — Black must play Kg8... but wait,
    // check Kh8 with Qf7 doesn't give check. Let's try a known single-move position.
    // Black king on a8, white queen on b6 — only Ka7 might be possible.
    // Better: use a discovered check position. Just verify the branch works.
    ChessEngine engine(100);
    // Black Kh8, White Qg7 is checkmate, not stalemate. Let's try:
    // Black Kh1, White Qg3 — Kh1 only legal move Kg1 or Kh2 (if available).
    // Simpler approach: use a position with one known legal move.
    // Rh1 + Kf2 vs Ka8: Black has only Ka7
    ChessBoard board("k7/8/1K6/8/8/8/8/8 b - - 0 1");
    // Ka8 can go to Ka7, Kb8, Kb7 — that's 3 moves, not 1. Need tighter constraint.
    // Let's use: Black Ka1, White Kb3, Rc2 — Black only has Kb1
    ChessBoard board2("8/8/8/8/8/1K6/2R5/k7 b - - 0 1");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board2, tc);
    // Just verify it returns a valid move; if only 1 legal move,
    // it returns immediately (depth=1, nodes=1)
    EXPECT_FALSE(result.best_move.uci().empty());
}

// --- ChessEngine: eval cache ---

TEST(ChessEngineEdge, EvalCacheHit) {
    ChessEngine engine(100);
    ChessBoard board;
    float eval1 = engine.evaluate(board);
    float eval2 = engine.evaluate(board);
    EXPECT_FLOAT_EQ(eval1, eval2);
}

TEST(ChessEngineEdge, EvalCacheDifferentPositions) {
    ChessEngine engine(100);
    ChessBoard board1;
    ChessBoard board2("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    float eval1 = engine.evaluate(board1);
    float eval2 = engine.evaluate(board2);
    EXPECT_NE(eval1, eval2);
}

// --- MCTS edge cases ---

TEST(MCTSEdge, MCTSGetBestMoveCheckmate) {
    MCTSEngine engine(100);
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_TRUE(result.best_move.uci().empty());
    EXPECT_FLOAT_EQ(result.score, -config::MATE_VALUE);
    EXPECT_EQ(result.depth, 0);
}

TEST(MCTSEdge, MCTSGetBestMoveStalemate) {
    MCTSEngine engine(100);
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_TRUE(result.best_move.uci().empty());
    EXPECT_FLOAT_EQ(result.score, 0.0f);
}

TEST(MCTSEdge, MCTSGetBestMoveSingleLegalMove) {
    MCTSEngine engine(100);
    // Position where few legal moves exist
    ChessBoard board("8/8/8/8/8/1K6/2R5/k7 b - - 0 1");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
}

TEST(MCTSEdge, MCTSEvaluateCheckmate) {
    MCTSEngine engine(100);
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    float eval = engine.evaluate(board);
    EXPECT_LT(eval, -100.0f);  // White is mated
}

TEST(MCTSEdge, MCTSSearchFromEndgame) {
    MCTSEngine engine(200);
    // Near-endgame position
    ChessBoard board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_FALSE(result.best_move.uci().empty());
    EXPECT_GT(result.nodes_searched, 0);
}

TEST(MCTSEdge, MCTSEvaluateStalemate) {
    MCTSEngine engine(100);
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    float eval = engine.evaluate(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

TEST(MCTSEdge, MCTSFindsMateInOne) {
    // White can play Qf7# (scholar's mate setup)
    MCTSEngine engine(500);
    ChessBoard board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4");
    TimeControl tc{60000, 0, 0};
    auto result = engine.get_best_move(board, tc);
    EXPECT_EQ(result.best_move.uci(), "h5f7");
}

// --- BaseEngine: raw_evaluate for black-checkmated position ---

TEST(BaseEngineEdge, EvaluateBlackCheckmated) {
    ChessEngine engine(100);
    // Back rank mate: black Kd8 is mated by Rd1#
    // Actually let's use a clear position: Kh8, white Qg7 is Qg7#? No.
    // Use: black Ka8, white Qb7 is checkmate (a-file trapped, b7 covers a8/a7/b8)
    // FEN: k7/1Q6/1K6/8/8/8/8/8 b - - 0 1
    // Ka8 in check from Qb7. Can king escape? a7 attacked by Q, b8 attacked by Q.
    // That is checkmate.
    ChessBoard board("k7/1Q6/1K6/8/8/8/8/8 b - - 0 1");
    float eval = engine.evaluate(board);
    // Black is mated, turn is black. raw_evaluate: board.turn()==BLACK, so returns
    // +MATE_VALUE
    EXPECT_FLOAT_EQ(eval, config::MATE_VALUE);
}

// --- BaseEngine: raw_evaluate with loaded NNUE model ---

TEST(BaseEngineEdge, EvaluateWithLoadedNNUE) {
    auto nnue = load_nnue_from_buffer();
    ASSERT_TRUE(nnue->is_loaded());

    ChessEngine engine(100, EvalMode::NNUE, nnue);
    ChessBoard board;
    float eval = engine.evaluate(board);
    EXPECT_TRUE(std::isfinite(eval));
    EXPECT_LE(std::abs(eval), config::MATE_VALUE);
}

TEST(BaseEngineEdge, EvaluateNNUECheckmatePosition) {
    auto nnue = load_nnue_from_buffer();
    ASSERT_TRUE(nnue->is_loaded());

    ChessEngine engine(100, EvalMode::NNUE, nnue);
    // White is checkmated (fool's mate) — raw_evaluate detects game over
    // Turn is white, white is mated => returns -MATE_VALUE
    ChessBoard mate("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    float eval = engine.evaluate(mate);
    EXPECT_FLOAT_EQ(eval, -config::MATE_VALUE);
}

// --- TrainingPosition struct size ---

TEST(SelfPlayEdge, TrainingPositionSize) { EXPECT_EQ(sizeof(TrainingPosition), 42u); }

// --- SelfPlay: en passant on a-file (file 0) ---

TEST(SelfPlayEdge, EncodePositionEnPassantAFile) {
    // White pawn on b5, black played a7a5 => ep on a6 (file 0)
    ChessBoard board("rnbqkbnr/1ppppppp/8/pP6/8/8/P1PPPPPP/RNBQKBNR w KQkq a6 0 3");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 4);
    EXPECT_EQ(pos.en_passant_file, 0);
}

TEST(SelfPlayEdge, EncodePositionEnPassantHFile) {
    // White pawn on g5, black played h7h5 => white to move, ep on h6 (file 7)
    // White g5 pawn can capture gxh6 en passant
    ChessBoard board("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq h6 0 3");
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 5);
    EXPECT_EQ(pos.en_passant_file, 7);
}

// --- ChessBoard: exact capture count ---

TEST(ChessBoardEdge, CaptureCountExact) {
    // After 1.e4 d5, white's only capture is exd5
    ChessBoard board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    auto captures = board.get_capture_moves();
    EXPECT_EQ(captures.size(), 1u);
    EXPECT_EQ(captures[0].uci(), "e4d5");
}

TEST(ChessBoardEdge, EnPassantCaptureIncludedInCaptures) {
    // White pawn on e5, black played d7d5 => exd6 en passant available
    ChessBoard board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    auto captures = board.get_capture_moves();
    bool found_ep = false;
    for (const auto& m : captures) {
        if (m.uci() == "e5d6") {
            found_ep = true;
            break;
        }
    }
    EXPECT_TRUE(found_ep) << "En passant capture should be in capture moves";
}

// --- ChessBoard: draw by insufficient material variants ---

TEST(ChessBoardEdge, KingBishopVsKingIsInsufficient) {
    ChessBoard board("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
    EXPECT_TRUE(board.is_draw());
    EXPECT_TRUE(board.is_game_over());
}

TEST(ChessBoardEdge, KingKnightVsKingIsInsufficient) {
    ChessBoard board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1");
    EXPECT_TRUE(board.is_draw());
    EXPECT_TRUE(board.is_game_over());
}

// --- ModelComparator::Result: minimal improvement ---

TEST(ModelComparatorEdge, ImprovedMinimal) {
    ModelComparator::Result result;
    result.new_wins = 1;
    result.old_wins = 0;
    result.draws = 99;
    EXPECT_TRUE(result.improved());
}

// --- BaseEngine: draw positions return 0 in raw_evaluate ---

TEST(BaseEngineEdge, EvaluateDrawByInsufficientMaterial) {
    ChessEngine engine(100);
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    float eval = engine.evaluate(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

TEST(BaseEngineEdge, EvaluateDrawByInsufficientMaterialNNUE) {
    // NNUE mode should also return 0 for drawn positions (raw_evaluate checks
    // game_over before calling NNUE)
    ChessEngine engine(100, EvalMode::NNUE, nullptr);
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    float eval = engine.evaluate(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

// --- SelfPlay: encode_position with all game results ---

TEST(SelfPlayEdge, EncodePositionWinResult) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 1000.0f, 2, 50);
    EXPECT_EQ(pos.game_result, 2);
}

TEST(SelfPlayEdge, EncodePositionDrawResult) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 50);
    EXPECT_EQ(pos.game_result, 1);
}

TEST(SelfPlayEdge, EncodePositionLossResult) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, -1000.0f, 0, 50);
    EXPECT_EQ(pos.game_result, 0);
}
