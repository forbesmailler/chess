#include "self_play.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

#include "chess_engine.h"
#include "handcrafted_eval.h"

SelfPlayGenerator::SelfPlayGenerator(const Config& config) : config(config) {}

uint8_t SelfPlayGenerator::encode_piece(const chess::Piece& piece) {
    if (piece == chess::Piece::NONE) return 0;

    // chess library: PieceType order is PAWN=0..KING=5, Color WHITE=0/BLACK=1
    int pt = static_cast<int>(piece.type());      // 0-5
    int color = static_cast<int>(piece.color());  // 0=white, 1=black
    return static_cast<uint8_t>(1 + pt + color * 6);
}

TrainingPosition SelfPlayGenerator::encode_position(const ChessBoard& board, float eval,
                                                    uint8_t result, uint16_t ply) {
    TrainingPosition pos;
    std::memset(&pos, 0, sizeof(pos));

    const auto& b = board.board;

    // Encode piece placement: 2 squares per byte (high nibble = even sq, low nibble = odd sq)
    for (int sq = 0; sq < 64; sq += 2) {
        uint8_t p0 = encode_piece(b.at(static_cast<chess::Square>(sq)));
        uint8_t p1 = encode_piece(b.at(static_cast<chess::Square>(sq + 1)));
        pos.piece_placement[sq / 2] = (p0 << 4) | (p1 & 0x0F);
    }

    pos.side_to_move = b.sideToMove() == chess::Color::WHITE ? 0 : 1;

    auto rights = board.get_castling_rights();
    pos.castling =
        static_cast<uint8_t>((rights.white_kingside << 3) | (rights.white_queenside << 2) |
                             (rights.black_kingside << 1) | rights.black_queenside);

    // En passant
    auto ep = b.enpassantSq();
    if (ep != chess::Square::NO_SQ) {
        pos.en_passant_file = static_cast<uint8_t>(ep.index() % 8);
    } else {
        pos.en_passant_file = 255;
    }

    pos.search_eval = eval;
    pos.game_result = result;
    pos.ply_number = ply;

    return pos;
}

void SelfPlayGenerator::write_position(std::ofstream& out, const TrainingPosition& pos) {
    out.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
}

bool SelfPlayGenerator::read_position(std::ifstream& in, TrainingPosition& pos) {
    in.read(reinterpret_cast<char*>(&pos), sizeof(pos));
    return in.good();
}

void SelfPlayGenerator::generate() {
    auto start = std::chrono::steady_clock::now();
    games_completed.store(0);
    total_positions.store(0);

    int threads = std::max(1, config.num_threads);
    int games_per_thread = config.num_games / threads;
    int remainder = config.num_games % threads;

    std::vector<std::thread> thread_pool;
    for (int t = 0; t < threads; ++t) {
        int n = games_per_thread + (t < remainder ? 1 : 0);
        if (n == 0) continue;
        thread_pool.emplace_back(&SelfPlayGenerator::play_games, this, n, config.output_file, t);
    }

    for (auto& t : thread_pool) t.join();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    std::cout << "Self-play complete: " << games_completed.load() << " games, "
              << total_positions.load() << " positions in " << elapsed.count() << "s" << std::endl;
}

void SelfPlayGenerator::play_games(int num_games, const std::string& output_file, int thread_id) {
    std::mt19937 rng(std::random_device{}() + thread_id);

    for (int g = 0; g < num_games; ++g) {
        std::vector<TrainingPosition> positions;
        positions.reserve(config.max_game_ply);

        auto engine = std::make_unique<ChessEngine>(config.search_time_ms);

        ChessBoard board;
        uint16_t ply = 0;
        int consecutive_resign = 0;
        int white_result = 1;

        while (ply < config.max_game_ply) {
            if (board.is_game_over()) {
                if (board.is_checkmate()) {
                    white_result = board.turn() == ChessBoard::WHITE ? 0 : 2;
                } else {
                    white_result = 1;
                }
                break;
            }

            float stm_eval;
            ChessBoard::Move chosen_move;

            if (ply < config.softmax_plies) {
                // Softmax move selection for opening diversity
                auto legal_moves = board.get_legal_moves();
                if (legal_moves.empty()) break;

                std::vector<float> scores(legal_moves.size());
                for (size_t i = 0; i < legal_moves.size(); ++i) {
                    ChessBoard copy = board;
                    copy.make_move(legal_moves[i]);
                    // Negate: evaluate returns from white's perspective,
                    // we want score for the side that just moved
                    float eval = engine->evaluate(copy);
                    scores[i] = board.turn() == ChessBoard::WHITE ? eval : -eval;
                }

                // Softmax with temperature
                float max_score = *std::max_element(scores.begin(), scores.end());
                std::vector<float> probs(scores.size());
                float sum = 0.0f;
                for (size_t i = 0; i < scores.size(); ++i) {
                    probs[i] = std::exp((scores[i] - max_score) / config.softmax_temperature);
                    sum += probs[i];
                }
                for (auto& p : probs) p /= sum;

                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int idx = dist(rng);
                chosen_move = legal_moves[idx];
                stm_eval = scores[idx];
            } else {
                // Normal best-move search
                TimeControl tc{60000, 0, 0};
                engine->set_max_time(config.search_time_ms);
                auto result = engine->get_best_move(board, tc);
                stm_eval = board.turn() == ChessBoard::WHITE ? result.score : -result.score;
                chosen_move = result.best_move;
            }

            positions.push_back(encode_position(board, stm_eval, 1, ply));

            // Resign adjudication
            if (std::abs(stm_eval) > config.resign_threshold) {
                consecutive_resign++;
                if (consecutive_resign >= config.resign_count) {
                    if (stm_eval < 0) {
                        white_result = board.turn() == ChessBoard::WHITE ? 0 : 2;
                    } else {
                        white_result = board.turn() == ChessBoard::WHITE ? 2 : 0;
                    }
                    break;
                }
            } else {
                consecutive_resign = 0;
            }

            if (chosen_move.uci().empty()) break;
            board.make_move(chosen_move);
            ply++;
        }

        // Fill in game results
        for (auto& pos : positions) {
            bool white_stm = pos.side_to_move == 0;
            if (white_result == 2) {
                pos.game_result = white_stm ? 2 : 0;
            } else if (white_result == 0) {
                pos.game_result = white_stm ? 0 : 2;
            } else {
                pos.game_result = 1;
            }
        }

        {
            std::lock_guard<std::mutex> lock(file_mutex);
            std::ofstream out(output_file, std::ios::binary | std::ios::app);
            for (const auto& pos : positions) {
                write_position(out, pos);
            }
        }

        int completed = games_completed.fetch_add(1) + 1;
        total_positions.fetch_add(static_cast<int>(positions.size()));

        if (completed % 10 == 0) {
            std::cout << "Self-play progress: " << completed << "/" << config.num_games << " games"
                      << std::endl;
        }
    }
}
