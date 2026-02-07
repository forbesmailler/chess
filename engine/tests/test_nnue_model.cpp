#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

#include "../chess_board.h"
#include "../nnue_model.h"

namespace {

// Create a small test weight file with known values
std::string create_test_weights(unsigned int seed = 42) {
    std::string path = "test_nnue_weights.bin";

    constexpr int INPUT = 768;
    constexpr int H1 = 256;
    constexpr int H2 = 32;
    constexpr int OUTPUT = 3;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    std::ofstream f(path, std::ios::binary);

    // Header
    f.write("NNUE", 4);
    uint32_t version = 1;
    uint32_t input_size = INPUT;
    uint32_t hidden1_size = H1;
    uint32_t hidden2_size = H2;
    uint32_t output_size = OUTPUT;
    f.write(reinterpret_cast<char*>(&version), 4);
    f.write(reinterpret_cast<char*>(&input_size), 4);
    f.write(reinterpret_cast<char*>(&hidden1_size), 4);
    f.write(reinterpret_cast<char*>(&hidden2_size), 4);
    f.write(reinterpret_cast<char*>(&output_size), 4);

    // Write random weights
    auto write_random = [&](size_t count) {
        for (size_t i = 0; i < count; ++i) {
            float val = dist(rng);
            f.write(reinterpret_cast<char*>(&val), sizeof(float));
        }
    };

    write_random(INPUT * H1);   // W1
    write_random(H1);           // b1
    write_random(H1 * H2);     // W2
    write_random(H2);           // b2
    write_random(H2 * OUTPUT);  // W3
    write_random(OUTPUT);       // b3

    return path;
}

}  // namespace

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
        f.write("XXXX", 4);  // Wrong magic
    }
    NNUEModel model;
    EXPECT_FALSE(model.load_weights(path));
    std::remove(path.c_str());
}

TEST(NNUEModel, PredictProducesFiniteOutput) {
    std::string path = create_test_weights();
    NNUEModel model;
    ASSERT_TRUE(model.load_weights(path));

    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_TRUE(std::isfinite(eval));
    std::remove(path.c_str());
}

TEST(NNUEModel, DifferentPositionsDifferentScores) {
    std::string path = create_test_weights();
    NNUEModel model;
    ASSERT_TRUE(model.load_weights(path));

    ChessBoard start_pos;
    float eval1 = model.predict(start_pos);

    // Position with different material
    ChessBoard board2("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1");
    float eval2 = model.predict(board2);

    // With random weights, these should almost certainly differ
    EXPECT_NE(eval1, eval2);
    std::remove(path.c_str());
}

TEST(NNUEModel, UnloadedModelReturnsZero) {
    NNUEModel model;
    ChessBoard board;
    float eval = model.predict(board);
    EXPECT_FLOAT_EQ(eval, 0.0f);
}

TEST(NNUEModel, CheckmateReturnsExtreme) {
    std::string path = create_test_weights();
    NNUEModel model;
    ASSERT_TRUE(model.load_weights(path));

    // Fool's mate - white is checkmated
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    float eval = model.predict(board);
    EXPECT_EQ(eval, -10000.0f);

    std::remove(path.c_str());
}
