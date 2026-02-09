#include <signal.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "base_engine.h"
#include "chess_board.h"
#include "chess_engine.h"
#include "lichess_client.h"
#include "mcts_engine.h"
#include "nnue_model.h"
#include "self_play.h"
#include "utils.h"
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

constexpr int MAX_RETRIES = config::bot::MAX_RETRIES;
constexpr int RETRY_DELAY_MS = config::bot::RETRY_DELAY_MS;
constexpr int HEARTBEAT_INTERVAL_MS = config::bot::HEARTBEAT_INTERVAL_MS;
constexpr int CONNECTION_TIMEOUT_MS = config::bot::CONNECTION_TIMEOUT_MS;
constexpr int MAX_CONSECUTIVE_ERRORS = config::bot::MAX_CONSECUTIVE_ERRORS;

// Global state for graceful shutdown
std::atomic<bool> shutdown_requested{false};
std::atomic<int> consecutive_errors{0};
std::atomic<std::chrono::steady_clock::time_point> last_activity{
    std::chrono::steady_clock::now()};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    Utils::log_info("Received shutdown signal " + std::to_string(signal) +
                    ", shutting down gracefully...");
    shutdown_requested.store(true);
}

struct GameState {
    ChessBoard board;
    int ply_count = 0;
    bool our_white = false;
    bool first_event = true;
    std::unique_ptr<BaseEngine> engine;
    std::chrono::steady_clock::time_point last_move_time;
    std::atomic<bool> is_active{true};
};

class LichessBot {
   public:
    enum class EngineType { NEGAMAX, MCTS };

    LichessBot(const std::string& token, int max_time_ms = 1000,
               EngineType engine_type = EngineType::NEGAMAX,
               EvalMode eval_mode = EvalMode::HANDCRAFTED,
               const std::string& nnue_weights_path = "")
        : client(token),
          engine_type(engine_type),
          eval_mode(eval_mode),
          heartbeat_active(true) {
        // Setup signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        // Load NNUE weights if requested
        if (eval_mode == EvalMode::NNUE && !nnue_weights_path.empty()) {
            nnue_model = std::make_shared<NNUEModel>();
            if (nnue_model->load_weights(nnue_weights_path)) {
                std::cout << "Loaded NNUE model: " << nnue_weights_path << std::endl;
            } else {
                Utils::log_warning("Failed to load NNUE weights from " +
                                   nnue_weights_path +
                                   ", falling back to handcrafted eval");
                this->eval_mode = EvalMode::HANDCRAFTED;
                nnue_model.reset();
            }
        }

        // Create engine based on type
        if (engine_type == EngineType::MCTS) {
            engine =
                std::make_unique<MCTSEngine>(max_time_ms, this->eval_mode, nnue_model);
            Utils::log_info("Using MCTS engine");
        } else {
            engine =
                std::make_unique<ChessEngine>(max_time_ms, this->eval_mode, nnue_model);
            Utils::log_info("Using Negamax engine");
        }

        std::string eval_name =
            this->eval_mode == EvalMode::NNUE ? "NNUE" : "Handcrafted";
        Utils::log_info("Using " + eval_name + " evaluation");

        if (!get_account_info_with_retry()) {
            throw std::runtime_error("Failed to get account information after retries");
        }

        Utils::log_info("Bot started as user: " + account_info.username + " (" +
                        account_info.id + ")");
        Utils::log_info("Max search time: " + std::to_string(max_time_ms) + "ms");

        if (account_info.is_bot) {
            Utils::log_info("Account is properly configured as a bot");
        } else {
            Utils::log_error("WARNING: Account is NOT configured as a bot! Title: " +
                             account_info.title);
            Utils::log_error("Please upgrade your account to a bot account on Lichess");
        }
    }

    ~LichessBot() {
        heartbeat_active = false;
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.join();
        }
    }

    void start() {
        Utils::log_info("Starting bot with error handling and keep-alive features...");

        // Start heartbeat monitor
        start_heartbeat_monitor();

        // Main bot loop with retry logic
        start_with_retry();
    }

   private:
    LichessClient client;
    std::shared_ptr<NNUEModel> nnue_model;
    std::unique_ptr<BaseEngine> engine;
    EngineType engine_type;
    EvalMode eval_mode;
    LichessClient::AccountInfo account_info;

    // Game state management
    std::unordered_map<std::string, std::shared_ptr<GameState>> active_games;
    std::mutex games_mutex;
    std::atomic<int> active_game_count{0};

    // Heartbeat and monitoring
    std::atomic<bool> heartbeat_active{true};
    std::thread heartbeat_thread;

    template <typename Func>
    bool retry_with_backoff(Func&& func, const std::string& action,
                            int delay_ms = RETRY_DELAY_MS) {
        for (int attempt = 1; attempt <= MAX_RETRIES; ++attempt) {
            try {
                if (func()) return true;
                Utils::log_warning("Failed: " + action + " (attempt " +
                                   std::to_string(attempt) + "/" +
                                   std::to_string(MAX_RETRIES) + ")");
            } catch (const std::exception& e) {
                Utils::log_error("Exception: " + action + " (attempt " +
                                 std::to_string(attempt) +
                                 "): " + std::string(e.what()));
            }
            if (attempt < MAX_RETRIES && !shutdown_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }
        }
        return false;
    }

    bool get_account_info_with_retry() {
        bool result = retry_with_backoff(
            [this]() { return client.get_account_info(account_info); },
            "get account info");
        if (result) consecutive_errors.store(0);
        return result;
    }

    void start_heartbeat_monitor() {
        heartbeat_thread = std::thread([this]() {
            while (heartbeat_active.load() && !shutdown_requested.load()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS));

                if (shutdown_requested.load()) break;

                // Check if we've been inactive too long
                auto now = std::chrono::steady_clock::now();
                auto time_since_activity = now - last_activity.load();

                if (std::chrono::duration_cast<std::chrono::milliseconds>(
                        time_since_activity)
                        .count() > CONNECTION_TIMEOUT_MS) {
                    Utils::log_warning(
                        "No activity for " +
                        std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                           time_since_activity)
                                           .count()) +
                        " seconds");

                    // Test basic connectivity first
                    if (client.test_connectivity()) {
                        // If basic connectivity is OK, test Lichess account info
                        LichessClient::AccountInfo test_info;
                        if (client.get_account_info(test_info)) {
                            Utils::log_info(
                                "Connection test passed - updating activity timestamp");
                            last_activity.store(now);
                            consecutive_errors.store(0);
                        } else {
                            Utils::log_error(
                                "Lichess account test failed, but basic connectivity "
                                "OK");
                            consecutive_errors.fetch_add(1);
                        }
                    } else {
                        Utils::log_error("Basic network connectivity failed");
                        consecutive_errors.fetch_add(1);
                    }
                }

                // Log status
                int error_count = consecutive_errors.load();
                int active_count = active_game_count.load();
                Utils::log_info("Heartbeat: " + std::to_string(active_count) +
                                " active games, " + std::to_string(error_count) +
                                " consecutive errors");

                // Check if we should shut down due to too many errors
                if (error_count > MAX_CONSECUTIVE_ERRORS) {
                    Utils::log_error("Too many consecutive errors (" +
                                     std::to_string(error_count) +
                                     "), requesting shutdown");
                    shutdown_requested.store(true);
                    break;
                }
            }
            Utils::log_info("Heartbeat monitor stopped");
        });
    }

    void start_with_retry() {
        int restart_attempts = 0;
        const int max_restarts = config::bot::MAX_RESTARTS;

        while (!shutdown_requested.load() && restart_attempts < max_restarts) {
            try {
                Utils::log_info("Starting event stream (attempt " +
                                std::to_string(restart_attempts + 1) + ")");
                consecutive_errors.store(0);

                stream_events_with_recovery();

                // If we get here, stream ended normally
                if (!shutdown_requested.load()) {
                    Utils::log_warning("Event stream ended unexpectedly, will retry");
                    restart_attempts++;
                }

            } catch (const std::exception& e) {
                Utils::log_error("Exception in main event loop: " +
                                 std::string(e.what()));
                restart_attempts++;
                consecutive_errors.fetch_add(1);
            }

            if (!shutdown_requested.load() && restart_attempts < max_restarts) {
                int delay =
                    RETRY_DELAY_MS * (restart_attempts + 1);  // Exponential backoff
                Utils::log_info("Restarting in " + std::to_string(delay / 1000) +
                                " seconds...");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));

                // Test connectivity before retrying
                Utils::log_info("Testing connectivity before retry...");
                if (!client.test_connectivity()) {
                    Utils::log_error("Connectivity test failed, waiting longer...");
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(delay));  // Wait extra time
                }
            }
        }

        if (restart_attempts >= max_restarts) {
            Utils::log_error("Maximum restart attempts reached, shutting down");
        }
    }

    void stream_events_with_recovery() {
        client.stream_events([this](const LichessClient::GameEvent& event) {
            if (shutdown_requested.load()) return;

            last_activity.store(std::chrono::steady_clock::now());

            try {
                handle_event(event);
                consecutive_errors.store(0);  // Reset on successful event handling
            } catch (const std::exception& e) {
                Utils::log_error("Error handling event: " + std::string(e.what()));
                consecutive_errors.fetch_add(1);

                if (consecutive_errors.load() > MAX_CONSECUTIVE_ERRORS) {
                    Utils::log_error(
                        "Too many consecutive errors, stopping event stream");
                    shutdown_requested.store(true);
                }
            }
        });
    }

    void handle_event(const LichessClient::GameEvent& event) {
        try {
            if (event.type == "challenge") {
                Utils::log_info("Received challenge: " + event.challenge_id);

                if (!accept_challenge_with_retry(event.challenge_id)) {
                    Utils::log_error("Failed to accept challenge after retries: " +
                                     event.challenge_id);
                } else {
                    Utils::log_info("Accepted challenge: " + event.challenge_id +
                                    " (Active games: " +
                                    std::to_string(active_game_count.load()) + ")");
                }

            } else if (event.type == "gameStart") {
                Utils::log_info("Game starting: " + event.game_id);

                std::lock_guard<std::mutex> lock(games_mutex);
                if (active_games.count(event.game_id)) {
                    Utils::log_info("Game " + event.game_id +
                                    " already has an active handler, skipping");
                    return;
                }
                auto game_state = std::make_shared<GameState>();

                // Create a separate engine instance for this game
                try {
                    if (engine_type == EngineType::MCTS) {
                        game_state->engine = std::make_unique<MCTSEngine>(
                            engine->get_max_time(), eval_mode, nnue_model);
                    } else {
                        game_state->engine = std::make_unique<ChessEngine>(
                            engine->get_max_time(), eval_mode, nnue_model);
                    }
                    game_state->last_move_time = std::chrono::steady_clock::now();
                    game_state->is_active.store(true);

                    active_games[event.game_id] = game_state;
                    active_game_count++;

                    Utils::log_info("Game started: " + event.game_id +
                                    " (Active games: " +
                                    std::to_string(active_game_count.load()) + ")");

                    // Start game handler in separate thread
                    std::thread game_thread(&LichessBot::handle_game_with_recovery,
                                            this, event.game_id);
                    game_thread.detach();

                } catch (const std::exception& e) {
                    Utils::log_error("Failed to initialize game " + event.game_id +
                                     ": " + std::string(e.what()));
                    active_games.erase(event.game_id);
                    active_game_count--;
                }

            } else {
                Utils::log_info("Ignoring event type: " + event.type);
            }
        } catch (const std::exception& e) {
            Utils::log_error("Exception in handle_event: " + std::string(e.what()));
            throw;  // Re-throw to be caught by caller
        }
    }

    bool accept_challenge_with_retry(const std::string& challenge_id) {
        return retry_with_backoff(
            [&]() { return client.accept_challenge(challenge_id); },
            "accept challenge " + challenge_id, RETRY_DELAY_MS / 2);
    }

    void handle_game_with_recovery(const std::string& game_id) {
        try {
            handle_game(game_id);
        } catch (const std::exception& e) {
            Utils::log_error("Fatal error in game " + game_id + ": " +
                             std::string(e.what()));
            cleanup_game(game_id);
        }
    }

    void handle_game(const std::string& game_id) {
        Utils::log_info("Starting game handler for: " + game_id);

        std::shared_ptr<GameState> game_state;
        {
            std::lock_guard<std::mutex> lock(games_mutex);
            auto it = active_games.find(game_id);
            if (it == active_games.end()) {
                Utils::log_error("Game state not found for game: " + game_id);
                return;
            }
            game_state = it->second;  // Keep shared_ptr alive
        }

        constexpr int MAX_STREAM_RETRIES = config::bot::MAX_GAME_STREAM_RETRIES;
        for (int attempt = 1; attempt <= MAX_STREAM_RETRIES; ++attempt) {
            if (shutdown_requested.load() || !game_state->is_active.load()) break;

            // Reset first_event so gameFull re-syncs board state on reconnect
            game_state->first_event = true;

            try {
                client.stream_game(game_id, [this, game_state, game_id](
                                                const LichessClient::GameEvent& event) {
                    if (shutdown_requested.load() || !game_state->is_active.load()) {
                        Utils::log_info(
                            "Game " + game_id +
                            ": Shutdown requested or game inactive, ending handler");
                        return;
                    }

                    last_activity.store(std::chrono::steady_clock::now());
                    game_state->last_move_time = std::chrono::steady_clock::now();

                    try {
                        handle_game_event(game_id, game_state, event);
                    } catch (const std::exception& e) {
                        Utils::log_error(
                            "Game " + game_id +
                            ": Error processing event: " + std::string(e.what()));
                        consecutive_errors.fetch_add(1);

                        if (consecutive_errors.load() > MAX_CONSECUTIVE_ERRORS / 2) {
                            Utils::log_error("Game " + game_id +
                                             ": Too many errors, abandoning game");
                            game_state->is_active.store(false);
                        }
                    }
                });
            } catch (const std::exception& e) {
                Utils::log_error("Game " + game_id + ": Exception in stream_game: " +
                                 std::string(e.what()));
            }

            // Stream ended — if game is still active, reconnect
            if (!game_state->is_active.load() || shutdown_requested.load()) break;

            if (attempt < MAX_STREAM_RETRIES) {
                Utils::log_warning("Game " + game_id +
                                   ": Stream disconnected, reconnecting (attempt " +
                                   std::to_string(attempt + 1) + "/" +
                                   std::to_string(MAX_STREAM_RETRIES) + ")");
                std::this_thread::sleep_for(
                    std::chrono::seconds(config::bot::GAME_STREAM_RECONNECT_DELAY_S));
            }
        }

        Utils::log_info("Game " + game_id + ": Stream ended, cleaning up");
        cleanup_game(game_id);
    }

    void handle_game_event(const std::string& game_id,
                           std::shared_ptr<GameState> game_state,
                           const LichessClient::GameEvent& event) {
        if (event.type == "gameFull" && game_state->first_event) {
            game_state->first_event = false;
            game_state->our_white = (event.white_id == account_info.id);
            Utils::log_info("Game " + game_id + ": we are " +
                            (game_state->our_white ? "White" : "Black"));
        } else if (event.type == "gameState") {
            if (event.status != "started") {
                Utils::log_info("Game " + game_id +
                                " ended with status: " + event.status);
                game_state->is_active.store(false);
                return;
            }
            if (event.draw_offer) handle_draw_offer_safely(game_id, game_state);
        } else {
            return;
        }

        process_moves(game_id, game_state, event.moves);

        float eval = evaluate_position_safely(game_state);
        Utils::log_info("Game " + game_id + " - Eval after ply " +
                        std::to_string(game_state->ply_count) +
                        " (white-persp): " + std::to_string(eval));

        if (is_our_turn(game_state)) {
            TimeControl time_control =
                create_time_control(event, game_state->our_white);
            play_best_move_safely(game_id, game_state, time_control);
        }
    }

    void process_moves(const std::string& game_id,
                       std::shared_ptr<GameState> game_state,
                       const std::string& moves_str) {
        auto moves = Utils::split_string(moves_str, ' ');

        for (size_t i = game_state->ply_count; i < moves.size(); i++) {
            const auto& uci = moves[i];
            if (uci.empty()) continue;

            try {
                auto move = ChessBoard::Move::from_uci(uci);
                game_state->board.make_move(move);
                game_state->ply_count++;

                // Log who made the move
                bool bot_move = ((i % 2 == 0 && game_state->our_white) ||
                                 (i % 2 == 1 && !game_state->our_white));
                std::string actor = bot_move ? "Bot" : "Opponent";
                Utils::log_info("Game " + game_id + ": " + actor + " played " + uci);

            } catch (const std::exception& e) {
                Utils::log_error("Game " + game_id + ": Invalid move '" + uci +
                                 "': " + std::string(e.what()));
                throw;  // This is a fatal error for the game
            }
        }
    }

    TimeControl create_time_control(const LichessClient::GameEvent& event,
                                    bool our_white) {
        int our_time = our_white ? event.wtime : event.btime;
        int our_increment = our_white ? event.winc : event.binc;

        Utils::log_info("Time control: " + std::to_string(our_time) + "ms + " +
                        std::to_string(our_increment) + "ms increment");

        return TimeControl{our_time, our_increment, 0};
    }

    bool play_best_move_safely(const std::string& game_id,
                               std::shared_ptr<GameState> game_state,
                               const TimeControl& time_control) {
        try {
            Utils::log_info("Game " + game_id + ": Thinking with " +
                            std::to_string(time_control.time_left_ms) + "ms left, " +
                            std::to_string(time_control.increment_ms) + "ms increment");

            auto search_result =
                game_state->engine->get_best_move(game_state->board, time_control);

            if (search_result.best_move.uci().empty()) {
                Utils::log_error("Game " + game_id + ": No valid move found!");
                return false;
            }

            Utils::log_info(
                "Game " + game_id + ": Found move " + search_result.best_move.uci() +
                " (depth: " + std::to_string(search_result.depth) +
                ", score: " + std::to_string(search_result.score) +
                ", time: " + std::to_string(search_result.time_used.count()) + "ms" +
                ", nodes: " + std::to_string(search_result.nodes_searched) + ")");

            // Try to make the move with retry logic
            if (make_move_with_retry(game_id, search_result.best_move.uci())) {
                Utils::log_info("Game " + game_id + ": Move sent successfully: " +
                                search_result.best_move.uci());
                game_state->board.make_move(search_result.best_move);
                game_state->ply_count++;
                game_state->last_move_time = std::chrono::steady_clock::now();
                return true;
            } else {
                Utils::log_error("Game " + game_id +
                                 ": Failed to send move after retries: " +
                                 search_result.best_move.uci());
                return false;
            }

        } catch (const std::exception& e) {
            Utils::log_error("Game " + game_id + ": Exception finding/playing move: " +
                             std::string(e.what()));
            return false;
        }
    }

    bool is_our_turn(std::shared_ptr<GameState> game_state) {
        return (game_state->board.turn() == ChessBoard::WHITE &&
                game_state->our_white) ||
               (game_state->board.turn() == ChessBoard::BLACK &&
                !game_state->our_white);
    }

    float evaluate_position_safely(std::shared_ptr<GameState> game_state) {
        try {
            return game_state->engine->evaluate(game_state->board);
        } catch (const std::exception& e) {
            Utils::log_warning("Error evaluating position: " + std::string(e.what()));
            return 0.0f;  // Return neutral evaluation on error
        }
    }

    bool make_move_with_retry(const std::string& game_id, const std::string& uci) {
        return retry_with_backoff([&]() { return client.make_move(game_id, uci); },
                                  "game " + game_id + " move " + uci,
                                  config::bot::MOVE_RETRY_DELAY_MS);
    }

    void handle_draw_offer_safely(const std::string& game_id,
                                  std::shared_ptr<GameState> game_state) {
        try {
            float eval = evaluate_position_safely(game_state);
            float our_eval = game_state->our_white ? eval : -eval;

            bool accept_draw = (our_eval <= 0.0f);

            Utils::log_info("Game " + game_id + " - Draw offer: " +
                            (accept_draw ? "ACCEPTING" : "DECLINING") +
                            " (our eval: " + std::to_string(our_eval) +
                            ", white eval: " + std::to_string(eval) + ")");

            // Try the draw response with simple retry
            for (int attempt = 1; attempt <= 2; ++attempt) {
                bool success = accept_draw ? client.accept_draw(game_id)
                                           : client.decline_draw(game_id);
                if (success) break;

                if (attempt == 1) {
                    Utils::log_warning("Game " + game_id +
                                       ": Draw response failed, retrying...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(
                        config::bot::DRAW_RESPONSE_RETRY_DELAY_MS));
                }
            }

        } catch (const std::exception& e) {
            Utils::log_error("Game " + game_id +
                             ": Error handling draw offer: " + std::string(e.what()));
        }
    }

    void cleanup_game(const std::string& game_id) {
        std::lock_guard<std::mutex> lock(games_mutex);
        auto it = active_games.find(game_id);
        if (it != active_games.end()) {
            // Mark game as inactive
            it->second->is_active.store(false);

            // Calculate game duration
            auto now = std::chrono::steady_clock::now();
            auto duration = now - it->second->last_move_time;
            auto duration_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(duration).count();

            active_games.erase(it);
            active_game_count--;

            Utils::log_info(
                "Game " + game_id + " cleaned up after " +
                std::to_string(duration_seconds) +
                " seconds. Active games: " + std::to_string(active_game_count.load()));
        }
    }
};

int main(int argc, char* argv[]) {
    // If no arguments provided, run tests
    if (argc == 1) {
        std::cout << "=== C++ Chess Bot Test Mode ===" << std::endl;
        std::cout << "Running chess board and engine tests..." << std::endl;
        std::cout << std::endl;

        ChessBoard board;
        std::cout << "Starting FEN: " << board.to_fen() << std::endl;

        auto moves = board.get_legal_moves();
        std::cout << "Legal moves from start: " << moves.size() << std::endl;

        if (!moves.empty()) {
            auto move = moves[0];
            std::cout << "Making move: " << move.uci() << std::endl;
            board.make_move(move);
            std::cout << "FEN after move: " << board.to_fen() << std::endl;
            board.unmake_move(move);
            std::cout << "FEN after unmake: " << board.to_fen() << std::endl;
        }

        std::cout << "Piece count: " << board.piece_count() << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << std::endl;
        std::cout << "Usage: " << argv[0] << " [max_time_ms] [options]" << std::endl;
        std::cout
            << "       " << argv[0]
            << " --selfplay [num_games] [search_depth] [output_file] [num_threads]"
               " [nnue_weights]"
            << std::endl;
        std::cout << "       " << argv[0]
                  << " --compare <old_weights|handcrafted> <new_weights>"
                     " [num_games] [output_file] [threads]"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Set LICHESS_TOKEN environment variable before running."
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --engine=negamax|mcts    Search algorithm (default: negamax)"
                  << std::endl;
        std::cout << "  --eval=handcrafted|nnue  Eval function (default: handcrafted)"
                  << std::endl;
        std::cout << "  --nnue-weights=<path>    Path to NNUE binary weights"
                  << std::endl;
        return 0;
    }

    // Check for --selfplay mode
    std::string first_arg = argv[1];
    if (first_arg == "--selfplay") {
        SelfPlayGenerator::Config config;
        if (argc > 2) config.num_games = std::stoi(argv[2]);
        if (argc > 3) config.search_depth = std::stoi(argv[3]);
        if (argc > 4) config.output_file = argv[4];
        if (argc > 5) config.num_threads = std::stoi(argv[5]);
        if (argc > 6) config.nnue_weights = argv[6];

        std::string eval_str = config.nnue_weights.empty()
                                   ? "handcrafted"
                                   : "NNUE (" + config.nnue_weights + ")";
        std::cout << "=== Self-Play Data Generation ===" << std::endl;
        std::cout << "Games: " << config.num_games << std::endl;
        std::cout << "Search depth (time budget): " << config.search_depth << std::endl;
        std::cout << "Output: " << config.output_file << std::endl;
        std::cout << "Threads: " << config.num_threads << std::endl;
        std::cout << "Eval: " << eval_str << std::endl;

        SelfPlayGenerator generator(config);
        generator.generate();
        std::cout << "Total positions: " << generator.get_total_positions()
                  << std::endl;
        return 0;
    }

    // Check for --compare mode
    if (first_arg == "--compare") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " --compare <old_weights|handcrafted> <new_weights>"
                         " [num_games] [output_file] [threads]"
                      << std::endl;
            return 1;
        }

        ModelComparator::Config config;
        std::string old_weights = argv[2];
        std::string new_weights = argv[3];
        if (argc > 4) config.num_games = std::stoi(argv[4]);
        if (argc > 5) config.output_file = argv[5];
        if (argc > 6) config.num_threads = std::stoi(argv[6]);

        // "handcrafted" means use handcrafted eval (empty string signals this)
        if (old_weights == "handcrafted") old_weights = "";

        std::cout << "=== Model Comparison ===" << std::endl;
        std::cout << "Old: " << (old_weights.empty() ? "handcrafted" : old_weights)
                  << std::endl;
        std::cout << "New: " << new_weights << std::endl;
        std::cout << "Games: " << config.num_games << std::endl;
        std::cout << "Output: "
                  << (config.output_file.empty() ? "(none)" : config.output_file)
                  << std::endl;
        std::cout << "Threads: " << config.num_threads << std::endl;

        ModelComparator comparator(config, old_weights, new_weights);
        auto result = comparator.run();

        std::cout << "Total positions generated: " << result.total_positions
                  << std::endl;
        if (result.improved()) {
            std::cout << "Result: NEW MODEL IMPROVED" << std::endl;
            return 0;
        } else {
            std::cout << "Result: No improvement" << std::endl;
            return 1;
        }
    }

    const char* token_env = std::getenv("LICHESS_TOKEN");
    if (!token_env || std::string(token_env).empty()) {
        std::cerr << "Error: LICHESS_TOKEN environment variable not set." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [max_time_ms] [options]" << std::endl;
        return 1;
    }

    std::string token = token_env;
    int max_time_ms = 1000;
    LichessBot::EngineType engine_type = LichessBot::EngineType::NEGAMAX;
    EvalMode eval_mode = EvalMode::HANDCRAFTED;
    std::string nnue_weights_path;

    // Parse optional arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg.find("--engine=") == 0) {
            std::string engine_str = arg.substr(9);
            if (engine_str == "mcts") {
                engine_type = LichessBot::EngineType::MCTS;
            } else if (engine_str == "negamax") {
                engine_type = LichessBot::EngineType::NEGAMAX;
            } else {
                std::cerr << "Unknown engine type: " << engine_str << ". Using negamax."
                          << std::endl;
            }
        } else if (arg.find("--eval=") == 0) {
            std::string eval_str = arg.substr(7);
            if (eval_str == "handcrafted") {
                eval_mode = EvalMode::HANDCRAFTED;
            } else if (eval_str == "nnue") {
                eval_mode = EvalMode::NNUE;
            } else {
                std::cerr << "Unknown eval mode: " << eval_str << ". Using handcrafted."
                          << std::endl;
            }
        } else if (arg.find("--nnue-weights=") == 0) {
            nnue_weights_path = arg.substr(15);
        } else {
            try {
                max_time_ms = std::stoi(arg);
                if (max_time_ms < config::search::MIN_TIME_MS ||
                    max_time_ms > config::search::MAX_TIME_MS) {
                    std::cerr << "Warning: Max time should be between "
                              << config::search::MIN_TIME_MS << "-"
                              << config::search::MAX_TIME_MS << "ms. Using "
                              << max_time_ms << "ms" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid parameter: " << arg << ", ignoring" << std::endl;
            }
        }
    }

    std::string engine_name =
        (engine_type == LichessBot::EngineType::MCTS) ? "MCTS" : "Negamax";
    std::string eval_name = eval_mode == EvalMode::NNUE ? "NNUE" : "Handcrafted";

    std::cout << "=== Starting Lichess Bot ===" << std::endl;
    std::cout << "Engine: " << engine_name << std::endl;
    std::cout << "Eval: " << eval_name << std::endl;
    std::cout << "Max search time: " << max_time_ms << "ms" << std::endl;
    std::cout << "Process ID: " << getpid() << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop the bot gracefully" << std::endl;
    std::cout << std::endl;

    int exit_code = 0;
    try {
        LichessBot bot(token, max_time_ms, engine_type, eval_mode, nnue_weights_path);
        Utils::log_info("Bot initialized successfully, starting main loop...");

        bot.start();

        if (shutdown_requested.load()) {
            Utils::log_info("Bot stopped gracefully");
        } else {
            Utils::log_warning("Bot stopped unexpectedly");
            exit_code = 1;
        }

    } catch (const std::exception& e) {
        Utils::log_error("Bot crashed with exception: " + std::string(e.what()));
        std::cerr << "Fatal error: " << e.what() << std::endl;
        exit_code = 1;
    }

    Utils::log_info("=== Bot Shutdown Complete ===");
    return exit_code;
}
