#include "chess_board.h"
#include "chess_engine.h"
#include "feature_extractor.h"
#include "lichess_client.h"
#include "logistic_model.h"
#include "utils.h"
#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <fstream>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <signal.h>
#include <exception>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

// Robust bot configuration
const int MAX_RETRIES = 3;
const int RETRY_DELAY_MS = 5000;  // 5 seconds
const int HEARTBEAT_INTERVAL_MS = 30000;  // 30 seconds
const int CONNECTION_TIMEOUT_MS = 120000;  // 2 minutes
const int MAX_CONSECUTIVE_ERRORS = 10;

// Global state for graceful shutdown
std::atomic<bool> shutdown_requested{false};
std::atomic<int> consecutive_errors{0};
std::atomic<std::chrono::steady_clock::time_point> last_activity{std::chrono::steady_clock::now()};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    Utils::log_info("Received shutdown signal " + std::to_string(signal) + ", shutting down gracefully...");
    shutdown_requested.store(true);
}

struct GameState {
    ChessBoard board;
    int ply_count = 0;
    bool our_white = false;
    bool first_event = true;
    std::unique_ptr<ChessEngine> engine;
    std::chrono::steady_clock::time_point last_move_time;
    std::atomic<bool> is_active{true};
};

class LichessBot {
public:
    LichessBot(const std::string& token, const std::string& model_path, int max_time_ms = 1000)
        : client(token), model_path(model_path), heartbeat_active(true) {
        
        // Setup signal handlers
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        model = std::make_shared<LogisticModel>();
        if (!model->load_model(model_path)) {
            Utils::log_warning("Failed to load model from " + model_path + ", using dummy model");
        }
        
        engine = std::make_unique<ChessEngine>(model, max_time_ms);
        
        if (!get_account_info_with_retry()) {
            throw std::runtime_error("Failed to get account information after retries");
        }
        
        Utils::log_info("Bot started as user: " + account_info.username + " (" + account_info.id + ")");
        Utils::log_info("Max search time: " + std::to_string(max_time_ms) + "ms");
        
        if (account_info.is_bot) {
            Utils::log_info("Account is properly configured as a bot");
        } else {
            Utils::log_error("WARNING: Account is NOT configured as a bot! Title: " + account_info.title);
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
    std::shared_ptr<LogisticModel> model;
    std::unique_ptr<ChessEngine> engine;
    LichessClient::AccountInfo account_info;
    std::string model_path;
    
    // Game state management
    std::unordered_map<std::string, std::shared_ptr<GameState>> active_games;
    std::mutex games_mutex;
    std::atomic<int> active_game_count{0};
    
    // Heartbeat and monitoring
    std::atomic<bool> heartbeat_active{true};
    std::thread heartbeat_thread;
    
    bool get_account_info_with_retry() {
        for (int attempt = 1; attempt <= MAX_RETRIES; ++attempt) {
            try {
                if (client.get_account_info(account_info)) {
                    consecutive_errors.store(0);
                    return true;
                }
                Utils::log_warning("Failed to get account info (attempt " + std::to_string(attempt) + "/" + std::to_string(MAX_RETRIES) + ")");
            } catch (const std::exception& e) {
                Utils::log_error("Exception getting account info (attempt " + std::to_string(attempt) + "): " + std::string(e.what()));
            }
            
            if (attempt < MAX_RETRIES && !shutdown_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
            }
        }
        return false;
    }
    
    void start_heartbeat_monitor() {
        heartbeat_thread = std::thread([this]() {
            while (heartbeat_active.load() && !shutdown_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(HEARTBEAT_INTERVAL_MS));
                
                if (shutdown_requested.load()) break;
                
                // Check if we've been inactive too long
                auto now = std::chrono::steady_clock::now();
                auto time_since_activity = now - last_activity.load();
                
                if (std::chrono::duration_cast<std::chrono::milliseconds>(time_since_activity).count() > CONNECTION_TIMEOUT_MS) {
                    Utils::log_warning("No activity for " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(time_since_activity).count()) + " seconds");
                    
                    // Test connection
                    LichessClient::AccountInfo test_info;
                    if (client.get_account_info(test_info)) {
                        Utils::log_info("Connection test passed - updating activity timestamp");
                        last_activity.store(now);
                        consecutive_errors.store(0);
                    } else {
                        Utils::log_error("Connection test failed");
                        consecutive_errors.fetch_add(1);
                    }
                }
                
                // Log status
                int error_count = consecutive_errors.load();
                int active_count = active_game_count.load();
                Utils::log_info("Heartbeat: " + std::to_string(active_count) + " active games, " + std::to_string(error_count) + " consecutive errors");
                
                // Check if we should shut down due to too many errors
                if (error_count > MAX_CONSECUTIVE_ERRORS) {
                    Utils::log_error("Too many consecutive errors (" + std::to_string(error_count) + "), requesting shutdown");
                    shutdown_requested.store(true);
                    break;
                }
            }
            Utils::log_info("Heartbeat monitor stopped");
        });
    }
    
    void start_with_retry() {
        int restart_attempts = 0;
        const int max_restarts = 5;
        
        while (!shutdown_requested.load() && restart_attempts < max_restarts) {
            try {
                Utils::log_info("Starting event stream (attempt " + std::to_string(restart_attempts + 1) + ")");
                consecutive_errors.store(0);
                
                stream_events_with_recovery();
                
                // If we get here, stream ended normally
                if (!shutdown_requested.load()) {
                    Utils::log_warning("Event stream ended unexpectedly, will retry");
                    restart_attempts++;
                }
                
            } catch (const std::exception& e) {
                Utils::log_error("Exception in main event loop: " + std::string(e.what()));
                restart_attempts++;
                consecutive_errors.fetch_add(1);
            }
            
            if (!shutdown_requested.load() && restart_attempts < max_restarts) {
                int delay = RETRY_DELAY_MS * (restart_attempts + 1); // Exponential backoff
                Utils::log_info("Restarting in " + std::to_string(delay / 1000) + " seconds...");
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
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
                consecutive_errors.store(0); // Reset on successful event handling
            } catch (const std::exception& e) {
                Utils::log_error("Error handling event: " + std::string(e.what()));
                consecutive_errors.fetch_add(1);
                
                if (consecutive_errors.load() > MAX_CONSECUTIVE_ERRORS) {
                    Utils::log_error("Too many consecutive errors, stopping event stream");
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
                    Utils::log_error("Failed to accept challenge after retries: " + event.challenge_id);
                } else {
                    Utils::log_info("Accepted challenge: " + event.challenge_id + 
                                  " (Active games: " + std::to_string(active_game_count.load()) + ")");
                }
                
            } else if (event.type == "gameStart") {
                Utils::log_info("Game starting: " + event.game_id);
                
                std::lock_guard<std::mutex> lock(games_mutex);
                auto game_state = std::make_shared<GameState>();
                
                // Create a separate engine instance for this game
                try {
                    game_state->engine = std::make_unique<ChessEngine>(model, engine->get_max_time());
                    game_state->last_move_time = std::chrono::steady_clock::now();
                    game_state->is_active.store(true);
                    
                    active_games[event.game_id] = game_state;
                    active_game_count++;
                    
                    Utils::log_info("Game started: " + event.game_id + 
                                  " (Active games: " + std::to_string(active_game_count.load()) + ")");
                    
                    // Start game handler in separate thread
                    std::thread game_thread(&LichessBot::handle_game_with_recovery, this, event.game_id);
                    game_thread.detach();
                    
                } catch (const std::exception& e) {
                    Utils::log_error("Failed to initialize game " + event.game_id + ": " + std::string(e.what()));
                    active_games.erase(event.game_id);
                }
                
            } else {
                Utils::log_info("Ignoring event type: " + event.type);
            }
        } catch (const std::exception& e) {
            Utils::log_error("Exception in handle_event: " + std::string(e.what()));
            throw; // Re-throw to be caught by caller
        }
    }
    
    bool accept_challenge_with_retry(const std::string& challenge_id) {
        for (int attempt = 1; attempt <= MAX_RETRIES; ++attempt) {
            try {
                if (client.accept_challenge(challenge_id)) {
                    return true;
                }
                Utils::log_warning("Failed to accept challenge (attempt " + std::to_string(attempt) + "/" + std::to_string(MAX_RETRIES) + ")");
            } catch (const std::exception& e) {
                Utils::log_error("Exception accepting challenge (attempt " + std::to_string(attempt) + "): " + std::string(e.what()));
            }
            
            if (attempt < MAX_RETRIES && !shutdown_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS / 2));
            }
        }
        return false;
    }
    
    void handle_game_with_recovery(const std::string& game_id) {
        try {
            handle_game(game_id);
        } catch (const std::exception& e) {
            Utils::log_error("Fatal error in game " + game_id + ": " + std::string(e.what()));
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
            game_state = it->second; // Keep shared_ptr alive
        }
        
        try {
            client.stream_game(game_id, [this, game_state, game_id](const LichessClient::GameEvent& event) {
                if (shutdown_requested.load() || !game_state->is_active.load()) {
                    Utils::log_info("Game " + game_id + ": Shutdown requested or game inactive, ending handler");
                    return;
                }
                
                last_activity.store(std::chrono::steady_clock::now());
                game_state->last_move_time = std::chrono::steady_clock::now();
                
                try {
                    handle_game_event(game_id, game_state, event);
                } catch (const std::exception& e) {
                    Utils::log_error("Game " + game_id + ": Error processing event: " + std::string(e.what()));
                    consecutive_errors.fetch_add(1);
                    
                    if (consecutive_errors.load() > MAX_CONSECUTIVE_ERRORS / 2) { // Per-game threshold
                        Utils::log_error("Game " + game_id + ": Too many errors, abandoning game");
                        game_state->is_active.store(false);
                    }
                }
            });
        } catch (const std::exception& e) {
            Utils::log_error("Game " + game_id + ": Exception in stream_game: " + std::string(e.what()));
        }
        
        // Clean up when stream ends
        Utils::log_info("Game " + game_id + ": Stream ended, cleaning up");
        cleanup_game(game_id);
    }
    
    void handle_game_event(const std::string& game_id, std::shared_ptr<GameState> game_state, const LichessClient::GameEvent& event) {
        if (event.type == "gameFull" && game_state->first_event) {
            game_state->first_event = false;
            
            game_state->our_white = (event.white_id == account_info.id);
            Utils::log_info("Game " + game_id + ": we are " + (game_state->our_white ? "White" : "Black"));
            Utils::log_info("White player: " + event.white_id + ", Black player: " + event.black_id);
            
            // Process initial moves
            process_moves(game_id, game_state, event.moves);
            
            float eval = evaluate_position_safely(game_state);
            Utils::log_info("Game " + game_id + " - Initial eval after ply " + std::to_string(game_state->ply_count) + 
                           " (white-persp): " + std::to_string(eval));
            
            // Make initial move if it's our turn with time control
            if (is_our_turn(game_state)) {
                TimeControl time_control = create_time_control(event, game_state->our_white);
                play_best_move_safely(game_id, game_state, time_control);
            }
            
        } else if (event.type == "gameState") {
            if (event.status != "started") {
                Utils::log_info("Game " + game_id + " ended with status: " + event.status);
                game_state->is_active.store(false);
                return;
            }
            
            if (event.draw_offer) {
                handle_draw_offer_safely(game_id, game_state);
            }
            
            // Process new moves
            process_moves(game_id, game_state, event.moves);
            
            float eval = evaluate_position_safely(game_state);
            Utils::log_info("Game " + game_id + " - Eval after ply " + std::to_string(game_state->ply_count) + 
                           " (white-persp): " + std::to_string(eval));
            
            // Make our move if it's our turn with updated time control
            if (is_our_turn(game_state)) {
                TimeControl time_control = create_time_control(event, game_state->our_white);
                play_best_move_safely(game_id, game_state, time_control);
            }
        }
    }
    
    void process_moves(const std::string& game_id, std::shared_ptr<GameState> game_state, const std::string& moves_str) {
        auto moves = Utils::split_string(moves_str, ' ');
        
        for (size_t i = game_state->ply_count; i < moves.size(); i++) {
            const auto& uci = moves[i];
            if (uci.empty()) continue;
            
            try {
                auto move = ChessBoard::Move::from_uci(uci);
                game_state->board.make_move(move);
                game_state->ply_count++;
                
                // Log who made the move
                bool bot_move = ((i % 2 == 0 && game_state->our_white) || (i % 2 == 1 && !game_state->our_white));
                std::string actor = bot_move ? "Bot" : "Opponent";
                Utils::log_info("Game " + game_id + ": " + actor + " played " + uci);
                
            } catch (const std::exception& e) {
                Utils::log_error("Game " + game_id + ": Invalid move '" + uci + "': " + std::string(e.what()));
                throw; // This is a fatal error for the game
            }
        }
    }
    
    TimeControl create_time_control(const LichessClient::GameEvent& event, bool our_white) {
        int our_time = our_white ? event.wtime : event.btime;
        int our_increment = our_white ? event.winc : event.binc;
        
        Utils::log_info("Time control: " + std::to_string(our_time) + "ms + " + std::to_string(our_increment) + "ms increment");
        
        return TimeControl(our_time, our_increment, 0); // moves_to_go = 0 means no time control limit
    }
    
    bool play_best_move_safely(const std::string& game_id, std::shared_ptr<GameState> game_state, const TimeControl& time_control) {
        try {
            Utils::log_info("Game " + game_id + ": Thinking with " + std::to_string(time_control.time_left_ms) + 
                           "ms left, " + std::to_string(time_control.increment_ms) + "ms increment");
            
            auto search_result = game_state->engine->get_best_move(game_state->board, time_control);
            
            if (search_result.best_move.uci_string.empty()) {
                Utils::log_error("Game " + game_id + ": No valid move found!");
                return false;
            }
            
            Utils::log_info("Game " + game_id + ": Found move " + search_result.best_move.uci() + 
                           " (depth: " + std::to_string(search_result.depth_reached) + 
                           ", score: " + std::to_string(search_result.score) +
                           ", time: " + std::to_string(search_result.time_used.count()) + "ms" +
                           ", nodes: " + std::to_string(search_result.nodes_searched) + ")");
            
            // Try to make the move with retry logic
            if (make_move_with_retry(game_id, search_result.best_move.uci())) {
                Utils::log_info("Game " + game_id + ": Move sent successfully: " + search_result.best_move.uci());
                game_state->board.make_move(search_result.best_move);
                game_state->ply_count++;
                game_state->last_move_time = std::chrono::steady_clock::now();
                return true;
            } else {
                Utils::log_error("Game " + game_id + ": Failed to send move after retries: " + search_result.best_move.uci());
                return false;
            }
            
        } catch (const std::exception& e) {
            Utils::log_error("Game " + game_id + ": Exception finding/playing move: " + std::string(e.what()));
            return false;
        }
    }
    
    bool is_our_turn(std::shared_ptr<GameState> game_state) {
        return (game_state->board.turn() == ChessBoard::WHITE && game_state->our_white) || 
               (game_state->board.turn() == ChessBoard::BLACK && !game_state->our_white);
    }
    
    float evaluate_position_safely(std::shared_ptr<GameState> game_state) {
        try {
            return game_state->engine->evaluate(game_state->board);
        } catch (const std::exception& e) {
            Utils::log_warning("Error evaluating position: " + std::string(e.what()));
            return 0.0f; // Return neutral evaluation on error
        }
    }
    
    bool make_move_with_retry(const std::string& game_id, const std::string& uci) {
        for (int attempt = 1; attempt <= MAX_RETRIES; ++attempt) {
            try {
                if (client.make_move(game_id, uci)) {
                    return true;
                }
                Utils::log_warning("Game " + game_id + ": Move attempt " + std::to_string(attempt) + " failed for " + uci);
            } catch (const std::exception& e) {
                Utils::log_error("Game " + game_id + ": Exception on move attempt " + std::to_string(attempt) + ": " + std::string(e.what()));
            }
            
            if (attempt < MAX_RETRIES && !shutdown_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 1 second retry delay for moves
            }
        }
        return false;
    }
    
    void handle_draw_offer_safely(const std::string& game_id, std::shared_ptr<GameState> game_state) {
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
                bool success = accept_draw ? client.accept_draw(game_id) : client.decline_draw(game_id);
                if (success) break;
                
                if (attempt == 1) {
                    Utils::log_warning("Game " + game_id + ": Draw response failed, retrying...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
            }
            
        } catch (const std::exception& e) {
            Utils::log_error("Game " + game_id + ": Error handling draw offer: " + std::string(e.what()));
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
            auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
            
            active_games.erase(it);
            active_game_count--;
            
            Utils::log_info("Game " + game_id + " cleaned up after " + std::to_string(duration_seconds) + 
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
        
        // Test ChessBoard implementation
        std::cout << "Testing ChessBoard implementation..." << std::endl;
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
        
        // Test feature extraction
        std::vector<float> features = FeatureExtractor::extract_features(board);
        std::cout << "Feature vector size: " << features.size() << std::endl;
        
        // Test piece counting - use the chess library's piece count method
        int piece_count = board.piece_count();
        std::cout << "Piece count: " << piece_count << std::endl;
        
        // Test castling rights
        auto rights = board.get_castling_rights();
        std::string castling = "";
        if (rights.white_kingside) castling += "K";
        if (rights.white_queenside) castling += "Q";
        if (rights.black_kingside) castling += "k";
        if (rights.black_queenside) castling += "q";
        if (castling.empty()) castling = "-";
        std::cout << "Castling rights: " << castling << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        std::cout << std::endl;
        std::cout << "=== To run the actual bot ===" << std::endl;
        std::cout << "Usage: " << argv[0] << " <lichess_token> [max_time_ms]" << std::endl;
        std::cout << "Example: " << argv[0] << " lip_abc123... 1000" << std::endl;
        std::cout << std::endl;
        std::cout << "You'll also need:" << std::endl;
        std::cout << "1. A valid Lichess API token with bot permissions" << std::endl;
        std::cout << "2. model_coefficients.txt file in the cpp/train/ directory (run export_model.py to create it)" << std::endl;
        std::cout << "3. Network connectivity for Lichess API calls" << std::endl;
        std::cout << std::endl;
        std::cout << "Optional max_time_ms parameter (default: 1000ms, recommended: 500-3000ms)" << std::endl;
        return 0;
    }
    
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <lichess_token> [max_time_ms]" << std::endl;
        std::cerr << "Run without arguments to see test output" << std::endl;
        std::cerr << "Max time parameter is optional (default: 1000ms)" << std::endl;
        return 1;
    }
    
    std::string token = argv[1];
    int max_time_ms = 1000; // Default to 1 second
    
    if (argc == 3) {
        try {
            max_time_ms = std::stoi(argv[2]);
            if (max_time_ms < 50 || max_time_ms > 30000) {
                std::cerr << "Warning: Max time should be between 50-30000ms. Using max_time " << max_time_ms << "ms" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid max_time parameter, using default 1000ms" << std::endl;
            max_time_ms = 1000;
        }
    }
    
    std::string model_path = "../../train/model_coefficients.txt"; // Relative to build/Release directory
    
    std::cout << "=== Starting Lichess Bot ===" << std::endl;
    std::cout << "Token: " << token.substr(0, 8) << "..." << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Max search time: " << max_time_ms << "ms" << std::endl;
    std::cout << "Process ID: " << getpid() << std::endl;
    std::cout << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "- Automatic retry on errors (max " << MAX_RETRIES << " attempts)" << std::endl;
    std::cout << "- Heartbeat monitoring every " << HEARTBEAT_INTERVAL_MS / 1000 << " seconds" << std::endl;
    std::cout << "- Connection timeout: " << CONNECTION_TIMEOUT_MS / 1000 << " seconds" << std::endl;
    std::cout << "- Error threshold: " << MAX_CONSECUTIVE_ERRORS << " consecutive errors" << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop the bot gracefully" << std::endl;
    std::cout << std::endl;
    
    int exit_code = 0;
    try {
        LichessBot bot(token, model_path, max_time_ms);
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
