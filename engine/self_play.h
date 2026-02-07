#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "chess_board.h"

// Binary format for training data: 42 bytes per position
struct TrainingPosition {
    uint8_t piece_placement[32];  // 64 nibbles (4 bits per square)
    uint8_t side_to_move;         // 0=white, 1=black
    uint8_t castling;             // 4 bits for KQkq
    uint8_t en_passant_file;      // 0-7 or 255 for none
    float search_eval;            // from side-to-move's perspective
    uint8_t game_result;          // 0=loss, 1=draw, 2=win (from STM perspective)
    uint16_t ply_number;
};

static_assert(sizeof(TrainingPosition) == 42 || sizeof(TrainingPosition) <= 48,
              "TrainingPosition should be compact");

class SelfPlayGenerator {
   public:
    struct Config {
        int num_games = 100;
        int search_depth = 6;
        int num_threads = 1;
        std::string output_file = "training_data.bin";
        int max_game_ply = 400;
        int search_time_ms = 200;     // time per move in milliseconds
        int resign_threshold = 5000;  // centipawns
        int resign_count = 3;         // consecutive moves above threshold
    };

    explicit SelfPlayGenerator(const Config& config);

    void generate();
    int get_total_positions() const { return total_positions; }

    // Encode/decode for binary format
    static TrainingPosition encode_position(const ChessBoard& board, float eval, uint8_t result,
                                            uint16_t ply);
    static void write_position(std::ofstream& out, const TrainingPosition& pos);
    static bool read_position(std::ifstream& in, TrainingPosition& pos);

   private:
    Config config;
    std::mutex file_mutex;
    std::atomic<int> games_completed{0};
    std::atomic<int> total_positions{0};

    void play_games(int num_games, const std::string& output_file);
    void play_single_game(std::ofstream& out);

    // Piece encoding: 0=empty, 1=wP, 2=wN, 3=wB, 4=wR, 5=wQ, 6=wK,
    //                 7=bP, 8=bN, 9=bB, 10=bR, 11=bQ, 12=bK
    static uint8_t encode_piece(const chess::Piece& piece);
};
