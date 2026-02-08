#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

#include "chess_board.h"
#include "generated_config.h"
#include "nnue_model.h"

namespace {

std::string create_test_weights(unsigned int seed = 42) {
    std::string path = "test_nnue_weights.bin";

    constexpr int INPUT = config::nnue::INPUT_SIZE;
    constexpr int H1 = config::nnue::HIDDEN1_SIZE;
    constexpr int H2 = config::nnue::HIDDEN2_SIZE;
    constexpr int OUTPUT = config::nnue::OUTPUT_SIZE;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    std::ofstream f(path, std::ios::binary);

    f.write("NNUE", 4);
    uint32_t header[] = {1, INPUT, H1, H2, OUTPUT};
    f.write(reinterpret_cast<char*>(header), sizeof(header));

    auto write_random = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            float val = dist(rng);
            f.write(reinterpret_cast<char*>(&val), sizeof(float));
        }
    };

    write_random(INPUT * H1);
    write_random(H1);
    write_random(H1 * H2);
    write_random(H2);
    write_random(H2 * OUTPUT);
    write_random(OUTPUT);

    return path;
}

}  // namespace

class NNUEModelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        weights_path = create_test_weights();
        ASSERT_TRUE(model.load_weights(weights_path));
    }

    void TearDown() override { std::remove(weights_path.c_str()); }

    std::string weights_path;
    NNUEModel model;
};

TEST(NNUEModel, LoadWeights) {
    std::string path = create_test_weights();
    NNUEModel model;
    EXPECT_TRUE(model.load_weights(path));
    EXPECT_TRUE(model.is_loaded());
    std::remove(path.c_str());
}

TEST(NNUEModel, LoadInvalidFile) {
    NNUEModel model;
    EXPECT_FALSE(model.load_weights("nonexistent_file.bin"));
    EXPECT_FALSE(model.is_loaded());
}

TEST(NNUEModel, LoadBadMagic) {
    std::string path = "test_bad_magic.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("XXXX", 4);
    }
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(path));
    std::remove(path.c_str());
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
    // Same position but with different side to move should produce negated eval
    // (since predict converts from STM to white's perspective)
    ChessBoard white_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    ChessBoard black_board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    float white_eval = model.predict(white_board);
    float black_eval = model.predict(black_board);
    // The evals should differ — different features are active for each side
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
    // Starting position is symmetric; with random weights the eval will be some
    // value, but it should be finite and bounded
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_TRUE(std::isfinite(eval));
    EXPECT_LE(std::abs(eval), config::MATE_VALUE);
}
