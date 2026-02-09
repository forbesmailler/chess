#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <sstream>

#include "chess_board.h"
#include "generated_config.h"
#include "nnue_model.h"

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
