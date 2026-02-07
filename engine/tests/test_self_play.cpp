#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>

#include "../chess_board.h"
#include "../self_play.h"

TEST(SelfPlay, EncodeDecodeRoundtrip) {
    ChessBoard board;
    float eval = 123.456f;
    uint8_t result = 2;  // win
    uint16_t ply = 42;

    TrainingPosition pos = SelfPlayGenerator::encode_position(board, eval, result, ply);

    EXPECT_EQ(pos.side_to_move, 0);  // White to move
    EXPECT_FLOAT_EQ(pos.search_eval, eval);
    EXPECT_EQ(pos.game_result, result);
    EXPECT_EQ(pos.ply_number, ply);
    EXPECT_EQ(pos.en_passant_file, 255);  // No en passant

    // Castling should have all 4 bits set
    EXPECT_EQ(pos.castling & 0x08, 0x08);  // White kingside
    EXPECT_EQ(pos.castling & 0x04, 0x04);  // White queenside
    EXPECT_EQ(pos.castling & 0x02, 0x02);  // Black kingside
    EXPECT_EQ(pos.castling & 0x01, 0x01);  // Black queenside
}

TEST(SelfPlay, BinaryWriteReadRoundtrip) {
    ChessBoard board;
    TrainingPosition pos = SelfPlayGenerator::encode_position(board, 50.0f, 1, 10);

    // Write to temp file
    std::string tmp_file = "test_roundtrip.bin";
    {
        std::ofstream out(tmp_file, std::ios::binary);
        SelfPlayGenerator::write_position(out, pos);
    }

    // Read back
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

    // Check that piece placement is non-trivial (not all zeros)
    bool has_pieces = false;
    for (int i = 0; i < 32; ++i) {
        if (pos.piece_placement[i] != 0) {
            has_pieces = true;
            break;
        }
    }
    EXPECT_TRUE(has_pieces);

    // Squares 16-47 (ranks 3-6) should be empty
    // sq 16 => byte 8 high nibble, sq 17 => byte 8 low nibble, etc.
    for (int sq = 16; sq < 48; sq += 2) {
        int byte_idx = sq / 2;
        EXPECT_EQ(pos.piece_placement[byte_idx], 0)
            << "Expected empty squares at byte " << byte_idx;
    }
}

TEST(SelfPlay, SingleGameProducesPositions) {
    std::string tmp_file = "test_single_game.bin";
    std::remove(tmp_file.c_str());

    SelfPlayGenerator::Config config;
    config.num_games = 1;
    config.num_threads = 1;
    config.output_file = tmp_file;
    config.search_time_ms = 10;
    config.max_game_ply = 50;
    config.resign_threshold = 500;

    SelfPlayGenerator generator(config);
    generator.generate();

    EXPECT_GT(generator.get_total_positions(), 0);

    // Verify file exists and has content
    std::ifstream in(tmp_file, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(in.is_open());
    auto file_size = in.tellg();
    EXPECT_GT(file_size, 0);

    // Each position should be sizeof(TrainingPosition) bytes
    EXPECT_EQ(file_size % sizeof(TrainingPosition), 0);

    std::remove(tmp_file.c_str());
}

TEST(SelfPlay, MultiThreadedNoCorruption) {
    std::string tmp_file = "test_multithread.bin";
    std::remove(tmp_file.c_str());

    SelfPlayGenerator::Config config;
    config.num_games = 4;
    config.num_threads = 2;
    config.output_file = tmp_file;
    config.search_time_ms = 10;
    config.max_game_ply = 50;
    config.resign_threshold = 500;

    SelfPlayGenerator generator(config);
    generator.generate();

    EXPECT_GT(generator.get_total_positions(), 0);

    // Verify file is valid: size should be multiple of position size
    std::ifstream in(tmp_file, std::ios::binary | std::ios::ate);
    auto file_size = in.tellg();
    EXPECT_EQ(file_size % sizeof(TrainingPosition), 0);

    // Read all positions and verify they're valid
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
