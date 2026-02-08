#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

#include "../chess_board.h"
#include "../nnue_model.h"

namespace {

std::string create_test_weights(unsigned int seed = 42) {
    std::string path = "test_nnue_weights.bin";

    constexpr int INPUT = 768;
    constexpr int H1 = 256;
    constexpr int H2 = 32;
    constexpr int OUTPUT = 3;

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

TEST_F(NNUEModelTest, CheckmateReturnsExtreme) {
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    float eval = model.predict(board);
    EXPECT_EQ(eval, -10000.0f);
}
