#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

#include "chess_board.h"
#include "generated_config.h"
#include "nnue_model.h"
#include "self_play.h"

namespace {

SelfPlayGenerator::Config make_test_config(const std::string& output_file,
                                           int num_games = 1, int num_threads = 1) {
    SelfPlayGenerator::Config config;
    config.num_games = num_games;
    config.num_threads = num_threads;
    config.output_file = output_file;
    config.search_time_ms = 50;
    config.max_game_ply = 50;
    config.resign_threshold = 5000;
    return config;
}

}  // namespace

TEST(SelfPlay, EncodeDecodeRoundtrip) {
    ChessBoard board;
    float eval = 123.456f;
    uint8_t result = 2;  // win
    uint16_t ply = 42;

    TrainingPosition pos = SelfPlayGenerator::encode_position(board, eval, result, ply);

    EXPECT_EQ(pos.side_to_move, 0);
    EXPECT_FLOAT_EQ(pos.search_eval, eval);
    EXPECT_EQ(pos.game_result, result);
    EXPECT_EQ(pos.ply_number, ply);
    EXPECT_EQ(pos.en_passant_file, 255);

    EXPECT_EQ(pos.castling & 0x08, 0x08);
    EXPECT_EQ(pos.castling & 0x04, 0x04);
    EXPECT_EQ(pos.castling & 0x02, 0x02);
    EXPECT_EQ(pos.castling & 0x01, 0x01);
}

TEST(SelfPlay, BinaryWriteReadRoundtrip) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 50.0f, 1, 10);

    std::string tmp_file = "test_roundtrip.bin";
    {
        std::ofstream out(tmp_file, std::ios::binary);
        SelfPlayGenerator::write_position(out, pos);
    }

    TrainingPosition read_pos;
    {
        std::ifstream in(tmp_file, std::ios::binary);
        EXPECT_TRUE(SelfPlayGenerator::read_position(in, read_pos));
    }

    EXPECT_EQ(std::memcmp(pos.piece_placement, read_pos.piece_placement, 32), 0);
    EXPECT_EQ(read_pos.side_to_move, pos.side_to_move);
    EXPECT_EQ(read_pos.castling, pos.castling);
    EXPECT_EQ(read_pos.en_passant_file, pos.en_passant_file);
    EXPECT_FLOAT_EQ(read_pos.search_eval, pos.search_eval);
    EXPECT_EQ(read_pos.game_result, pos.game_result);
    EXPECT_EQ(read_pos.ply_number, pos.ply_number);

    std::remove(tmp_file.c_str());
}

TEST(SelfPlay, PieceEncodingStartPosition) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 0.0f, 1, 0);

    bool has_pieces = false;
    for (int i = 0; i < 32; ++i) {
        if (pos.piece_placement[i] != 0) {
            has_pieces = true;
            break;
        }
    }
    EXPECT_TRUE(has_pieces);

    for (int sq = 16; sq < 48; sq += 2) {
        int byte_idx = sq / 2;
        EXPECT_EQ(pos.piece_placement[byte_idx], 0)
            << "Expected empty squares at byte " << byte_idx;
    }
}

TEST(SelfPlay, SingleGameProducesPositions) {
    std::string tmp_file = "test_single_game.bin";
    std::remove(tmp_file.c_str());

    auto config = make_test_config(tmp_file);
    SelfPlayGenerator generator(config);
    generator.generate();

    EXPECT_GT(generator.get_total_positions(), 0);

    std::ifstream in(tmp_file, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(in.is_open());
    auto file_size = in.tellg();
    EXPECT_GT(file_size, 0);
    EXPECT_EQ(file_size % sizeof(TrainingPosition), 0);

    std::remove(tmp_file.c_str());
}

TEST(SelfPlay, MultiThreadedNoCorruption) {
    std::string tmp_file = "test_multithread.bin";
    std::remove(tmp_file.c_str());

    auto config = make_test_config(tmp_file, 4, 2);
    SelfPlayGenerator generator(config);
    generator.generate();

    EXPECT_GT(generator.get_total_positions(), 0);

    std::ifstream in(tmp_file, std::ios::binary | std::ios::ate);
    auto file_size = in.tellg();
    EXPECT_EQ(file_size % sizeof(TrainingPosition), 0);

    in.seekg(0);
    int count = 0;
    TrainingPosition pos;
    while (SelfPlayGenerator::read_position(in, pos)) {
        EXPECT_LE(pos.side_to_move, 1);
        EXPECT_LE(pos.game_result, 2);
        EXPECT_TRUE(std::isfinite(pos.search_eval));
        count++;
    }
    EXPECT_EQ(count, generator.get_total_positions());

    std::remove(tmp_file.c_str());
}

TEST(SelfPlay, SoftmaxProducesDiverseGames) {
    // Run two games with softmax enabled — positions should differ
    std::string file1 = "test_diversity_1.bin";
    std::string file2 = "test_diversity_2.bin";
    std::remove(file1.c_str());
    std::remove(file2.c_str());

    auto config1 = make_test_config(file1);
    config1.softmax_plies = 10;
    config1.softmax_temperature = 200.0f;
    SelfPlayGenerator gen1(config1);
    gen1.generate();

    auto config2 = make_test_config(file2);
    config2.softmax_plies = 10;
    config2.softmax_temperature = 200.0f;
    SelfPlayGenerator gen2(config2);
    gen2.generate();

    // Read positions from both games and compare piece placements at ply 5+
    auto read_positions = [](const std::string& file) {
        std::vector<TrainingPosition> positions;
        std::ifstream in(file, std::ios::binary);
        TrainingPosition pos;
        while (SelfPlayGenerator::read_position(in, pos)) {
            positions.push_back(pos);
        }
        return positions;
    };

    auto pos1 = read_positions(file1);
    auto pos2 = read_positions(file2);

    // Both games should have produced positions
    ASSERT_GT(pos1.size(), 5u);
    ASSERT_GT(pos2.size(), 5u);

    // At least one position after ply 3 should differ (with overwhelming probability)
    size_t compare_count = std::min(pos1.size(), pos2.size());
    bool found_difference = false;
    for (size_t i = 3; i < compare_count; ++i) {
        if (std::memcmp(pos1[i].piece_placement, pos2[i].piece_placement, 32) != 0) {
            found_difference = true;
            break;
        }
    }
    EXPECT_TRUE(found_difference) << "Two softmax games produced identical positions";

    std::remove(file1.c_str());
    std::remove(file2.c_str());
}

namespace {

std::string create_compare_weights(const std::string& path, unsigned int seed) {
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

TEST(ModelComparator, ComparisonProducesResults) {
    std::string old_weights = create_compare_weights("test_cmp_old.bin", 42);
    std::string new_weights = create_compare_weights("test_cmp_new.bin", 123);
    std::string output_file = "test_cmp_output.bin";
    std::remove(output_file.c_str());

    ModelComparator::Config config;
    config.num_games = 2;
    config.num_threads = 1;
    config.output_file = output_file;
    config.max_game_ply = 50;
    config.search_time_ms = 50;

    ModelComparator comparator(config, old_weights, new_weights);
    auto result = comparator.run();

    EXPECT_EQ(result.new_wins + result.old_wins + result.draws, 2);
    EXPECT_GT(result.total_positions, 0);

    // Verify binary output is valid
    std::ifstream in(output_file, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(in.is_open());
    auto file_size = in.tellg();
    EXPECT_GT(file_size, 0);
    EXPECT_EQ(file_size % sizeof(TrainingPosition), 0);

    std::remove(old_weights.c_str());
    std::remove(new_weights.c_str());
    std::remove(output_file.c_str());
}
