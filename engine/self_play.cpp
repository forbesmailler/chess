#include "self_play.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>

#include "chess_engine.h"
#include "nnue_model.h"
#include "opening_book.h"

SelfPlayGenerator::SelfPlayGenerator(const Config& config) : config(config) {}

uint8_t SelfPlayGenerator::encode_piece(const chess::Piece& piece) {
    if (piece == chess::Piece::NONE) return 0;

    int pt = static_cast<int>(piece.type());
    int color = static_cast<int>(piece.color());
    return static_cast<uint8_t>(1 + pt + color * 6);
}

TrainingPosition SelfPlayGenerator::encode_position(const ChessBoard& board, float eval,
                                                    uint8_t result, uint16_t ply) {
    TrainingPosition pos;
    std::memset(&pos, 0, sizeof(pos));

    const auto& b = board.board;

    for (int sq = 0; sq < 64; sq += 2) {
        uint8_t p0 = encode_piece(b.at(static_cast<chess::Square>(sq)));
        uint8_t p1 = encode_piece(b.at(static_cast<chess::Square>(sq + 1)));
        pos.piece_placement[sq / 2] = (p0 << 4) | (p1 & 0x0F);
    }

    pos.side_to_move = b.sideToMove() == chess::Color::WHITE ? 0 : 1;

    auto rights = board.get_castling_rights();
    pos.castling =
        static_cast<uint8_t>((static_cast<int>(rights.white_kingside) << 3) |
                             (static_cast<int>(rights.white_queenside) << 2) |
                             (static_cast<int>(rights.black_kingside) << 1) |
                             static_cast<int>(rights.black_queenside));

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

void SelfPlayGenerator::write_position(std::ofstream& out,
                                       const TrainingPosition& pos) {
    out.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
}

bool SelfPlayGenerator::read_position(std::ifstream& in, TrainingPosition& pos) {
    in.read(reinterpret_cast<char*>(&pos), sizeof(pos));
    return in.good();
}

ChessBoard SelfPlayGenerator::decode_position(const TrainingPosition& pos) {
    // Nibble → FEN char: 0=empty, 1-6=PNBRQK, 7-12=pnbrqk
    static const char NIBBLE_CHAR[] = {'\0', 'P', 'N', 'B', 'R', 'Q', 'K',
                                       'p',  'n', 'b', 'r', 'q', 'k'};

    std::string fen;
    fen.reserve(80);

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            uint8_t byte = pos.piece_placement[sq / 2];
            uint8_t nibble = (sq % 2 == 0) ? (byte >> 4) : (byte & 0x0F);
            if (nibble == 0) {
                ++empty;
            } else {
                if (empty > 0) {
                    fen += static_cast<char>('0' + empty);
                    empty = 0;
                }
                fen += NIBBLE_CHAR[nibble];
            }
        }
        if (empty > 0) fen += static_cast<char>('0' + empty);
        if (rank > 0) fen += '/';
    }

    fen += (pos.side_to_move == 0) ? " w " : " b ";

    // Castling: bit3=K, bit2=Q, bit1=k, bit0=q
    std::string castling;
    if (pos.castling & 0x08) castling += 'K';
    if (pos.castling & 0x04) castling += 'Q';
    if (pos.castling & 0x02) castling += 'k';
    if (pos.castling & 0x01) castling += 'q';
    fen += castling.empty() ? "-" : castling;

    // En passant
    if (pos.en_passant_file < 8) {
        fen += ' ';
        fen += static_cast<char>('a' + pos.en_passant_file);
        fen += (pos.side_to_move == 0) ? '6' : '3';
    } else {
        fen += " -";
    }

    fen += " 0 ";
    fen += std::to_string(pos.ply_number / 2 + 1);

    return ChessBoard(fen);
}

void relabel_data(const std::string& input_file, const std::string& nnue_weights,
                  const std::string& output_file) {
    auto model = std::make_shared<NNUEModel>();
    if (!model->load_weights(nnue_weights)) {
        std::cerr << "Failed to load NNUE weights: " << nnue_weights << std::endl;
        return;
    }
    std::cout << "Loaded NNUE model: " << nnue_weights << std::endl;

    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open input: " << input_file << std::endl;
        return;
    }

    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output: " << output_file << std::endl;
        return;
    }

    TrainingPosition pos;
    int64_t count = 0;
    auto start = std::chrono::steady_clock::now();

    while (SelfPlayGenerator::read_position(in, pos)) {
        ChessBoard board = SelfPlayGenerator::decode_position(pos);
        float white_eval = model->predict(board);
        pos.search_eval = (pos.side_to_move == 0) ? white_eval : -white_eval;
        SelfPlayGenerator::write_position(out, pos);
        ++count;
        if (count % 1000000 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start);
            std::cout << "Relabeled " << count << " positions (" << elapsed.count()
                      << "s)" << std::endl;
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "Relabel complete: " << count << " positions in " << elapsed.count()
              << "s" << std::endl;
}

void SelfPlayGenerator::generate() {
    auto start = std::chrono::steady_clock::now();
    start_time = start;
    games_completed.store(0);
    total_positions.store(0);

    if (!config.book_path.empty()) {
        book_ = std::make_shared<OpeningBook>();
        if (book_->load(config.book_path)) {
            std::cout << "Loaded opening book: " << config.book_path << " ("
                      << book_->size() << " positions)" << std::endl;
        } else {
            std::cerr << "Failed to load opening book: " << config.book_path
                      << std::endl;
            book_.reset();
        }
    }

    int threads = std::max(1, config.num_threads);
    int games_per_thread = config.num_games / threads;
    int remainder = config.num_games % threads;

    std::vector<std::thread> thread_pool;
    for (int t = 0; t < threads; ++t) {
        int n = games_per_thread + (t < remainder ? 1 : 0);
        if (n == 0) continue;
        thread_pool.emplace_back(&SelfPlayGenerator::play_games, this, n,
                                 config.output_file, t);
    }

    for (auto& t : thread_pool) t.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    std::cout << "Self-play complete: " << games_completed.load() << " games, "
              << total_positions.load() << " positions in " << elapsed.count() << "s"
              << std::endl;
}

void SelfPlayGenerator::play_games(int num_games, const std::string& output_file,
                                   int thread_id) {
    std::mt19937 rng(std::random_device{}() + thread_id);

    std::shared_ptr<NNUEModel> model;
    EvalMode eval_mode = EvalMode::HANDCRAFTED;
    if (!config.nnue_weights.empty()) {
        model = std::make_shared<NNUEModel>();
        if (model->load_weights(config.nnue_weights)) {
            eval_mode = EvalMode::NNUE;
        } else {
            std::cerr << "Thread " << thread_id
                      << ": Failed to load NNUE weights, using handcrafted"
                      << std::endl;
            model = nullptr;
        }
    }

    auto engine =
        std::make_unique<ChessEngine>(config.search_time_ms, eval_mode, model);

    for (int g = 0; g < num_games; ++g) {
        engine->clear_caches();
        std::vector<TrainingPosition> positions;
        positions.reserve(config.max_game_ply);

        ChessBoard board;
        uint16_t ply = 0;
        int white_result = 1;

        while (ply < config.max_game_ply) {
            auto [reason, result] = board.board.isGameOver();
            if (result != chess::GameResult::NONE) {
                if (reason == chess::GameResultReason::CHECKMATE) {
                    white_result = board.turn() == ChessBoard::WHITE ? 0 : 2;
                } else {
                    white_result = 1;
                }
                break;
            }

            float stm_eval;
            ChessBoard::Move chosen_move;
            bool white_to_move = board.turn() == ChessBoard::WHITE;
            bool move_found = false;

            if (book_ && book_->is_loaded()) {
                auto book_move = book_->probe(board.board);
                if (book_move) {
                    chosen_move =
                        ChessBoard::Move::from_uci(chess::uci::moveToUci(*book_move));
                    stm_eval = engine->evaluate(board);
                    if (!white_to_move) stm_eval = -stm_eval;
                    move_found = true;
                }
            }

            if (!move_found && ply < config.random_plies) {
                // Fully random moves for opening diversity
                auto legal_moves = board.get_legal_moves();
                if (legal_moves.empty()) break;
                std::uniform_int_distribution<int> move_dist(
                    0, static_cast<int>(legal_moves.size()) - 1);
                chosen_move = legal_moves[move_dist(rng)];
                stm_eval = engine->evaluate(board);
                if (!white_to_move) stm_eval = -stm_eval;
            } else if (!move_found && ply < config.softmax_plies &&
                       config.softmax_temperature > 0.0f) {
                auto legal_moves = board.get_legal_moves();
                if (legal_moves.empty()) break;

                std::vector<float> scores(legal_moves.size());
                for (size_t i = 0; i < legal_moves.size(); ++i) {
                    board.board.makeMove(legal_moves[i].internal_move);
                    float eval = engine->evaluate(board);
                    scores[i] = white_to_move ? eval : -eval;
                    board.board.unmakeMove(legal_moves[i].internal_move);
                }

                float max_score = *std::max_element(scores.begin(), scores.end());
                std::vector<float> probs(scores.size());
                float sum = 0.0f;
                for (size_t i = 0; i < scores.size(); ++i) {
                    probs[i] =
                        std::exp((scores[i] - max_score) / config.softmax_temperature);
                    sum += probs[i];
                }
                for (auto& p : probs) p /= sum;

                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                int idx = dist(rng);
                chosen_move = legal_moves[idx];
                stm_eval = scores[idx];
            } else if (!move_found) {
                TimeControl tc{0, 0, 0};
                engine->set_max_time(config.search_time_ms);
                auto result = engine->get_best_move(board, tc);
                stm_eval = result.score;
                chosen_move = result.best_move;
            }

            positions.push_back(encode_position(board, stm_eval, 1, ply));

            if (chosen_move.uci().empty()) break;
            board.make_move(chosen_move);
            ply++;
        }

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

        std::string result_str = white_result == 2   ? "white wins"
                                 : white_result == 0 ? "black wins"
                                                     : "draw";
        std::cout << "Game " << completed << "/" << config.num_games << ": " << ply
                  << " plies, result=" << result_str << std::endl;

        if (completed % config::self_play::PROGRESS_LOG_INTERVAL == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            std::cout << "Self-play progress: " << completed << "/" << config.num_games
                      << " games, " << total_positions.load() << " positions, "
                      << elapsed.count() << "s elapsed" << std::endl;
        }
    }
}

ModelComparator::ModelComparator(const Config& config, const std::string& old_weights,
                                 const std::string& new_weights)
    : config(config), old_weights_path(old_weights), new_weights_path(new_weights) {}

ModelComparator::ModelComparator(const Config& config,
                                 std::shared_ptr<NNUEModel> old_model,
                                 std::shared_ptr<NNUEModel> new_model)
    : config(config),
      preloaded_old_model(std::move(old_model)),
      preloaded_new_model(std::move(new_model)) {}

ModelComparator::Result ModelComparator::run() {
    start_time = std::chrono::steady_clock::now();
    games_completed.store(0);
    new_wins.store(0);
    old_wins.store(0);
    draws.store(0);
    total_positions.store(0);

    if (!config.book_path.empty()) {
        book_ = std::make_shared<OpeningBook>();
        if (book_->load(config.book_path)) {
            std::cout << "Loaded opening book: " << config.book_path << " ("
                      << book_->size() << " positions)" << std::endl;
        } else {
            std::cerr << "Failed to load opening book: " << config.book_path
                      << std::endl;
            book_.reset();
        }
    }

    int threads = std::max(1, config.num_threads);
    int games_per_thread = config.num_games / threads;
    int remainder = config.num_games % threads;

    std::cout << "Loading models on " << threads << " thread(s)..." << std::endl;
    std::vector<std::vector<TaggedPosition>> thread_positions(threads);
    std::vector<std::thread> thread_pool;
    for (int t = 0; t < threads; ++t) {
        int n = games_per_thread + (t < remainder ? 1 : 0);
        if (n == 0) continue;
        thread_pool.emplace_back(&ModelComparator::play_games, this, n, t,
                                 std::ref(thread_positions[t]));
    }

    for (auto& t : thread_pool) t.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    std::cout << "Comparison complete: " << games_completed.load() << " games in "
              << elapsed.count() << "s" << std::endl;
    std::cout << "New wins: " << new_wins.load() << ", Old wins: " << old_wins.load()
              << ", Draws: " << draws.load() << std::endl;

    Result result;
    result.new_wins = new_wins.load();
    result.old_wins = old_wins.load();
    result.draws = draws.load();
    result.total_positions = total_positions.load();

    // Write positions from the overall winning engine only
    if (!config.output_file.empty() && result.new_wins != result.old_wins) {
        bool new_won_overall = result.new_wins > result.old_wins;
        std::ofstream out(config.output_file, std::ios::binary | std::ios::app);
        int written = 0;
        for (auto& tvec : thread_positions) {
            for (auto& tp : tvec) {
                if (tp.from_new_engine == new_won_overall) {
                    SelfPlayGenerator::write_position(out, tp.pos);
                    written++;
                }
            }
        }
        std::cout << "Wrote " << written << " positions from "
                  << (new_won_overall ? "new" : "old") << " engine to "
                  << config.output_file << std::endl;
    }

    return result;
}

void ModelComparator::play_games(int num_games, int thread_id,
                                 std::vector<TaggedPosition>& out_positions) {
    std::shared_ptr<NNUEModel> new_model = preloaded_new_model;
    if (!new_model) {
        new_model = std::make_shared<NNUEModel>();
        if (!new_model->load_weights(new_weights_path)) {
            std::cerr << "Thread " << thread_id << ": Failed to load new weights"
                      << std::endl;
            return;
        }
    }

    std::shared_ptr<NNUEModel> old_model = preloaded_old_model;
    bool old_is_handcrafted = !old_model && old_weights_path.empty();
    if (!old_model && !old_is_handcrafted) {
        old_model = std::make_shared<NNUEModel>();
        if (!old_model->load_weights(old_weights_path)) {
            std::cerr << "Thread " << thread_id << ": Failed to load old weights"
                      << std::endl;
            return;
        }
    }

    auto new_engine =
        std::make_unique<ChessEngine>(config.search_time_ms, EvalMode::NNUE, new_model);
    std::unique_ptr<ChessEngine> old_engine;
    if (old_is_handcrafted) {
        old_engine = std::make_unique<ChessEngine>(config.search_time_ms);
    } else {
        old_engine = std::make_unique<ChessEngine>(config.search_time_ms,
                                                   EvalMode::NNUE, old_model);
    }

    for (int g = 0; g < num_games; ++g) {
        new_engine->clear_caches();
        old_engine->clear_caches();
        std::vector<TrainingPosition> positions;
        std::vector<bool> from_new_engine;
        positions.reserve(config.max_game_ply);
        from_new_engine.reserve(config.max_game_ply);

        bool new_is_white = (thread_id + g) % 2 == 0;

        ChessBoard board;
        uint16_t ply = 0;
        int white_result = 1;

        while (ply < config.max_game_ply) {
            auto [go_reason, go_result] = board.board.isGameOver();
            if (go_result != chess::GameResult::NONE) {
                if (go_reason == chess::GameResultReason::CHECKMATE) {
                    white_result = board.turn() == ChessBoard::WHITE ? 0 : 2;
                } else {
                    white_result = 1;
                }
                break;
            }

            bool white_to_move = board.turn() == ChessBoard::WHITE;
            bool new_to_move = (white_to_move == new_is_white);
            ChessEngine* active = new_to_move ? new_engine.get() : old_engine.get();

            ChessBoard::Move chosen_move;
            float stm_eval;
            bool book_used = false;

            if (book_ && book_->is_loaded()) {
                auto book_move = book_->probe(board.board);
                if (book_move) {
                    chosen_move =
                        ChessBoard::Move::from_uci(chess::uci::moveToUci(*book_move));
                    stm_eval = active->evaluate(board);
                    if (!white_to_move) stm_eval = -stm_eval;
                    book_used = true;
                }
            }

            if (!book_used) {
                TimeControl tc{0, 0, 0};
                active->set_max_time(config.search_time_ms);
                auto result = active->get_best_move(board, tc);
                stm_eval = result.score;
                chosen_move = result.best_move;
            }

            positions.push_back(
                SelfPlayGenerator::encode_position(board, stm_eval, 1, ply));
            from_new_engine.push_back(new_to_move);

            if (chosen_move.uci().empty()) break;
            board.make_move(chosen_move);
            ply++;
        }

        bool new_won =
            (new_is_white && white_result == 2) || (!new_is_white && white_result == 0);
        bool old_won =
            (new_is_white && white_result == 0) || (!new_is_white && white_result == 2);

        for (size_t i = 0; i < positions.size(); ++i) {
            auto& pos = positions[i];
            bool white_stm = pos.side_to_move == 0;
            if (white_result == 2) {
                pos.game_result = white_stm ? 2 : 0;
            } else if (white_result == 0) {
                pos.game_result = white_stm ? 0 : 2;
            } else {
                pos.game_result = 1;
            }
        }

        // Collect positions with engine tags for deferred writing
        for (size_t i = 0; i < positions.size(); ++i) {
            out_positions.push_back({positions[i], from_new_engine[i]});
        }

        if (new_won)
            new_wins.fetch_add(1);
        else if (old_won)
            old_wins.fetch_add(1);
        else
            draws.fetch_add(1);

        int completed = games_completed.fetch_add(1) + 1;
        total_positions.fetch_add(static_cast<int>(positions.size()));

        std::string result_str = new_won ? "new wins" : old_won ? "old wins" : "draw";
        std::string color_str = new_is_white ? "new as white" : "new as black";
        std::cout << "Compare game " << completed << "/" << config.num_games << ": "
                  << ply << " plies, " << result_str << " (" << color_str << ")"
                  << std::endl;

        if (completed % config::self_play::PROGRESS_LOG_INTERVAL == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            std::cout << "Compare progress: " << completed << "/" << config.num_games
                      << " games, " << total_positions.load() << " positions, "
                      << elapsed.count() << "s elapsed" << std::endl;
        }
    }
}
