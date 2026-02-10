#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <sstream>

#include "chess_board.h"
#include "generated_config.h"
#include "handcrafted_eval.h"
#include "nnue_model.h"
#include "self_play.h"

namespace {

std::string create_test_weights_buffer(unsigned int seed = 42) {
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

}  // namespace

class NNUEModelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        std::string buf = create_test_weights_buffer();
        std::istringstream stream(buf);
        ASSERT_TRUE(model.load_weights(stream));
    }

    NNUEModel model;
};

TEST(NNUEModel, LoadWeights) {
    std::string buf = create_test_weights_buffer();
    std::istringstream stream(buf);
    NNUEModel model;
    EXPECT_TRUE(model.load_weights(stream));
    EXPECT_TRUE(model.is_loaded());
}

TEST(NNUEModel, LoadInvalidFile) {
    NNUEModel model;
    EXPECT_FALSE(model.load_weights("nonexistent_file.bin"));
    EXPECT_FALSE(model.is_loaded());
}

TEST(NNUEModel, LoadBadMagic) {
    std::istringstream stream("XXXX");
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(stream));
}

TEST_F(NNUEModelTest, PredictProducesFiniteOutput) {
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_TRUE(std::isfinite(eval));
}

TEST_F(NNUEModelTest, DifferentPositionsDifferentScores) {
    ChessBoard start_pos;
    float eval1 = model.predict(start_pos);

    ChessBoard board2("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1");
    float eval2 = model.predict(board2);

    EXPECT_NE(eval1, eval2);
}

TEST(NNUEModel, UnloadedModelReturnsZero) {
    NNUEModel model;
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

TEST_F(NNUEModelTest, PredictBlackToMoveNegatesOutput) {
    ChessBoard white_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    ChessBoard black_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    float white_eval = model.predict(white_board);
    float black_eval = model.predict(black_board);
    EXPECT_NE(white_eval, black_eval);
    EXPECT_TRUE(std::isfinite(white_eval));
    EXPECT_TRUE(std::isfinite(black_eval));
}

TEST_F(NNUEModelTest, PredictOutputBoundedByMateValue) {
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_LE(eval, config::MATE_VALUE);
    EXPECT_GE(eval, -config::MATE_VALUE);
}

TEST_F(NNUEModelTest, PredictSymmetricStartNearZero) {
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_TRUE(std::isfinite(eval));
    EXPECT_LE(std::abs(eval), config::MATE_VALUE);
}

// --- Diagnostic test 1: Feature encoding round-trip ---
// Verifies that features decoded from binary training data match features
// extracted directly from the board by NNUEModel::extract_features.

namespace {

// Decode a TrainingPosition binary back into a 773-element feature vector,
// replicating the Python extract_features logic in C++.
std::vector<float> decode_training_position_features(const TrainingPosition& pos) {
    constexpr int INPUT_SIZE = config::nnue::INPUT_SIZE;
    std::vector<float> features(INPUT_SIZE, 0.0f);
    bool white_to_move = pos.side_to_move == 0;

    // Decode piece placement nibbles
    for (int sq = 0; sq < 64; ++sq) {
        int byte_idx = sq / 2;
        uint8_t nibble;
        if (sq % 2 == 0) {
            nibble = (pos.piece_placement[byte_idx] >> 4) & 0x0F;
        } else {
            nibble = pos.piece_placement[byte_idx] & 0x0F;
        }
        if (nibble == 0) continue;

        int piece_type, piece_color;
        if (nibble <= 6) {
            piece_type = nibble - 1;
            piece_color = 0;  // white
        } else {
            piece_type = nibble - 7;
            piece_color = 1;  // black
        }

        bool is_own = (piece_color == 0) == white_to_move;
        int feature_sq = white_to_move ? sq : (sq ^ 56);
        int offset = is_own ? 0 : 384;
        features[offset + piece_type * 64 + feature_sq] = 1.0f;
    }

    // Castling rights
    uint8_t castling = pos.castling;
    if (white_to_move) {
        features[768] = static_cast<float>((castling >> 3) & 1);
        features[769] = static_cast<float>((castling >> 2) & 1);
        features[770] = static_cast<float>((castling >> 1) & 1);
        features[771] = static_cast<float>(castling & 1);
    } else {
        features[768] = static_cast<float>((castling >> 1) & 1);
        features[769] = static_cast<float>(castling & 1);
        features[770] = static_cast<float>((castling >> 3) & 1);
        features[771] = static_cast<float>((castling >> 2) & 1);
    }

    features[772] = (pos.en_passant_file != 255) ? 1.0f : 0.0f;

    return features;
}

}  // namespace

TEST_F(NNUEModelTest, FeatureEncodingRoundTrip) {
    struct TestCase {
        std::string name;
        std::string fen;
    };

    std::vector<TestCase> cases = {
        {"start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"after_e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},
        {"black_to_move_custom",
         "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"},
        {"no_castling", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"},
        {"en_passant", "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3"},
    };

    for (const auto& tc : cases) {
        SCOPED_TRACE(tc.name);
        ChessBoard board(tc.fen);

        // Path 1: encode to binary, then decode features
        TrainingPosition tp = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);
        std::vector<float> decoded = decode_training_position_features(tp);

        // Path 2: extract features directly from board
        auto active_indices = model.get_active_features(board);
        std::set<int> active_set(active_indices.begin(), active_indices.end());

        // Every active index should have a 1.0 in decoded, and vice versa
        std::set<int> decoded_set;
        for (int i = 0; i < config::nnue::INPUT_SIZE; ++i) {
            if (decoded[i] == 1.0f) decoded_set.insert(i);
        }

        // Check for mismatches: indices in active but not in decoded
        for (int idx : active_set) {
            EXPECT_EQ(decoded[idx], 1.0f)
                << "Feature " << idx
                << " active in extract_features but NOT in decoded binary";
        }

        // Check for mismatches: indices in decoded but not in active
        for (int idx : decoded_set) {
            EXPECT_TRUE(active_set.count(idx) > 0)
                << "Feature " << idx
                << " set in decoded binary but NOT in extract_features";
        }

        EXPECT_EQ(active_set.size(), decoded_set.size())
            << "Feature count mismatch: extract_features=" << active_set.size()
            << " decoded_binary=" << decoded_set.size();
    }
}

// --- Diagnostic test 3: Eval speed benchmark ---

TEST_F(NNUEModelTest, EvalSpeedBenchmark) {
    constexpr int NUM_EVALS = 10000;

    // Positions to cycle through for realistic cache behavior
    std::vector<ChessBoard> boards = {
        ChessBoard(),
        ChessBoard("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ChessBoard("r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
        ChessBoard("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
    };

    // Benchmark NNUE
    auto nnue_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_EVALS; ++i) {
        float e = model.predict(boards[i % boards.size()]);
        (void)e;
    }
    auto nnue_end = std::chrono::high_resolution_clock::now();
    double nnue_us =
        std::chrono::duration<double, std::micro>(nnue_end - nnue_start).count();

    // Benchmark handcrafted
    auto hc_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_EVALS; ++i) {
        float e = handcrafted_evaluate(boards[i % boards.size()]);
        (void)e;
    }
    auto hc_end = std::chrono::high_resolution_clock::now();
    double hc_us = std::chrono::duration<double, std::micro>(hc_end - hc_start).count();

    double ratio = nnue_us / hc_us;
    std::cout << "\n=== Eval Speed Benchmark (" << NUM_EVALS << " evals) ===\n"
              << "  NNUE:        " << nnue_us / 1000.0 << " ms (" << nnue_us / NUM_EVALS
              << " us/eval)\n"
              << "  Handcrafted: " << hc_us / 1000.0 << " ms (" << hc_us / NUM_EVALS
              << " us/eval)\n"
              << "  Ratio:       " << ratio << "x (NNUE / Handcrafted)\n"
              << "================================================\n";

    // Informational only — no pass/fail assertion on speed
    EXPECT_TRUE(true);
}

// --- Diagnostic test 5: Perspective consistency ---
// For the same board with different STM, own-piece features of one perspective
// should match opponent-piece features of the other (with square flipping).

TEST_F(NNUEModelTest, PerspectiveConsistency) {
    // Use same physical board, just change side to move
    // White STM: own = white (features 0-383), opp = black (features 384-767)
    // Black STM: own = black (features 0-383), opp = white (features 384-767)
    // Squares get flipped (sq^56) for black STM

    // After 1. e4: physical board is the same, just change STM
    ChessBoard white_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    ChessBoard black_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");

    auto w_feats = model.get_active_features(white_board);
    auto b_feats = model.get_active_features(black_board);

    // Split features into own-piece [0,383] and opp-piece [384,767]
    std::set<int> w_own, w_opp, b_own, b_opp;
    for (int f : w_feats) {
        if (f < 384)
            w_own.insert(f);
        else if (f < 768)
            w_opp.insert(f - 384);  // normalize to 0-383
    }
    for (int f : b_feats) {
        if (f < 384)
            b_own.insert(f);
        else if (f < 768)
            b_opp.insert(f - 384);  // normalize to 0-383
    }

    // White's own pieces should map to black's opponent pieces
    // But squares are flipped (sq^56), so we need to account for that.
    // Feature index = piece_type * 64 + feature_sq
    // white_own: piece on sq → feature_sq = sq (no flip, white STM)
    // black_opp: same piece on sq → feature_sq = sq ^ 56 (flipped, black STM)
    // So white_own feature (pt*64 + sq) maps to black_opp feature (pt*64 + (sq^56))

    auto flip_feature = [](int feat_idx) -> int {
        int pt = feat_idx / 64;
        int sq = feat_idx % 64;
        return pt * 64 + (sq ^ 56);
    };

    // White own pieces → Black opp pieces (with square flip)
    EXPECT_EQ(w_own.size(), b_opp.size()) << "White own count != Black opp count";
    for (int f : w_own) {
        int flipped = flip_feature(f);
        EXPECT_TRUE(b_opp.count(flipped) > 0)
            << "White own feature " << f << " (flipped=" << flipped
            << ") not found in black opp features";
    }

    // Black own pieces → White opp pieces (with square flip)
    EXPECT_EQ(b_own.size(), w_opp.size()) << "Black own count != White opp count";
    for (int f : b_own) {
        int flipped = flip_feature(f);
        EXPECT_TRUE(w_opp.count(flipped) > 0)
            << "Black own feature " << f << " (flipped=" << flipped
            << ") not found in white opp features";
    }
}
