#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "chess_board.h"
#include "generated_config.h"

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
        int num_games = config::self_play::NUM_GAMES;
        int search_depth = config::self_play::SEARCH_DEPTH;
        int num_threads = config::self_play::NUM_THREADS;
        std::string output_file = "training_data.bin";
        int max_game_ply = config::self_play::MAX_GAME_PLY;
        int search_time_ms = config::self_play::SEARCH_TIME_MS;
        int resign_threshold = config::self_play::RESIGN_THRESHOLD;
        int resign_count = config::self_play::RESIGN_COUNT;
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
