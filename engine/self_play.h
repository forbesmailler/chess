#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "chess_board.h"
#include "generated_config.h"

class NNUEModel;

#pragma pack(push, 1)
struct TrainingPosition {
    uint8_t piece_placement[32];  // 64 nibbles (4 bits per square)
    uint8_t side_to_move;         // 0=white, 1=black
    uint8_t castling;             // 4 bits for KQkq
    uint8_t en_passant_file;      // 0-7 or 255 for none
    float search_eval;            // from side-to-move's perspective
    uint8_t game_result;          // 0=loss, 1=draw, 2=win (from STM perspective)
    uint16_t ply_number;
};
#pragma pack(pop)

static_assert(sizeof(TrainingPosition) == 42,
              "TrainingPosition must be exactly 42 bytes");

class SelfPlayGenerator {
   public:
    struct Config {
        int num_games = config::self_play::NUM_GAMES;
        int search_depth = config::self_play::SEARCH_DEPTH;
        int num_threads = config::self_play::NUM_THREADS;
        std::string output_file = "training_data.bin";
        int max_game_ply = 400;
        int search_time_ms = config::self_play::SEARCH_TIME_MS;
        int random_plies = config::self_play::RANDOM_PLIES;
        int softmax_plies = config::self_play::SOFTMAX_PLIES;
        float softmax_temperature = config::self_play::SOFTMAX_TEMPERATURE;
        std::string nnue_weights;
    };

    explicit SelfPlayGenerator(const Config& config);

    void generate();
    int get_total_positions() const { return total_positions; }

    static TrainingPosition encode_position(const ChessBoard& board, float eval,
                                            uint8_t result, uint16_t ply);
    static ChessBoard decode_position(const TrainingPosition& pos);
    static void write_position(std::ofstream& out, const TrainingPosition& pos);
    static bool read_position(std::ifstream& in, TrainingPosition& pos);

   private:
    Config config;
    std::mutex file_mutex;
    std::atomic<int> games_completed{0};
    std::atomic<int> total_positions{0};
    std::chrono::steady_clock::time_point start_time;

    void play_games(int num_games, const std::string& output_file, int thread_id);

    // Piece encoding: 0=empty, 1=wP, 2=wN, 3=wB, 4=wR, 5=wQ, 6=wK,
    //                 7=bP, 8=bN, 9=bB, 10=bR, 11=bQ, 12=bK
    static uint8_t encode_piece(const chess::Piece& piece);
};

void relabel_data(const std::string& input_file, const std::string& nnue_weights,
                  const std::string& output_file);

class ModelComparator {
   public:
    struct Config {
        int num_games = config::compare::NUM_GAMES;
        int num_threads = config::self_play::NUM_THREADS;
        std::string output_file;
        int max_game_ply = 400;
        int search_time_ms = config::self_play::SEARCH_TIME_MS;
    };

    struct Result {
        int new_wins = 0, old_wins = 0, draws = 0, total_positions = 0;
        bool improved() const { return new_wins > old_wins; }
    };

    ModelComparator(const Config& config, const std::string& old_weights,
                    const std::string& new_weights);

    ModelComparator(const Config& config, std::shared_ptr<NNUEModel> old_model,
                    std::shared_ptr<NNUEModel> new_model);

    Result run();

   private:
    Config config;
    std::string old_weights_path;
    std::string new_weights_path;
    std::shared_ptr<NNUEModel> preloaded_old_model;
    std::shared_ptr<NNUEModel> preloaded_new_model;
    std::atomic<int> games_completed{0};
    std::atomic<int> new_wins{0};
    std::atomic<int> old_wins{0};
    std::atomic<int> draws{0};
    std::atomic<int> total_positions{0};
    std::chrono::steady_clock::time_point start_time;

    struct TaggedPosition {
        TrainingPosition pos;
        bool from_new_engine;
    };

    void play_games(int num_games, int thread_id,
                    std::vector<TaggedPosition>& out_positions);
};
