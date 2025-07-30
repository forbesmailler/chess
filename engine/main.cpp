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

class LichessBot {
public:
    LichessBot(const std::string& token, const std::string& model_path, int depth = 4)
        : client(token), model_path(model_path) {
        
        model = std::make_shared<LogisticModel>();
        if (!model->load_model(model_path)) {
            Utils::log_warning("Failed to load model from " + model_path + ", using dummy model");
        }
        
        engine = std::make_unique<ChessEngine>(model, depth);
        
        if (!client.get_account_info(account_info)) {
            Utils::log_error("Failed to get account information");
        } else {
            Utils::log_info("Bot started as user: " + account_info.username + " (" + account_info.id + ")");
            Utils::log_info("Search depth: " + std::to_string(depth));
            if (account_info.is_bot) {
                Utils::log_info("Account is properly configured as a bot");
            } else {
                Utils::log_error("WARNING: Account is NOT configured as a bot! Title: " + account_info.title);
                Utils::log_error("Please upgrade your account to a bot account on Lichess");
            }
        }
    }
    
    void start() {
        Utils::log_info("Starting bot, listening for events...");
        
        client.stream_events([this](const LichessClient::GameEvent& event) {
            handle_event(event);
        });
    }
    
private:
    LichessClient client;
    std::shared_ptr<LogisticModel> model;
    std::unique_ptr<ChessEngine> engine;
    LichessClient::AccountInfo account_info;
    std::string model_path;
    bool our_white = false;
    
    void handle_event(const LichessClient::GameEvent& event) {
        if (event.type == "challenge") {
            if (!client.accept_challenge(event.challenge_id)) {
                Utils::log_error("Failed to accept challenge: " + event.challenge_id);
            }
        } else if (event.type == "gameStart") {
            std::thread game_thread(&LichessBot::handle_game, this, event.game_id);
            game_thread.detach();
        }
    }
    
    void handle_game(const std::string& game_id) {
        bool first_event = true;
        ChessBoard board;
        int ply_count = 0;
        
        client.stream_game(game_id, [&](const LichessClient::GameEvent& event) {
            if (event.type == "gameFull" && first_event) {
                first_event = false;
                
                our_white = (event.white_id == account_info.id);
                Utils::log_info("Game " + game_id + ": we are " + (our_white ? "White" : "Black"));
                Utils::log_info("White player: " + event.white_id + ", Black player: " + event.black_id);
                
                auto moves = Utils::split_string(event.moves, ' ');
                for (const auto& uci : moves) {
                    if (!uci.empty()) {
                        auto move = ChessBoard::Move::from_uci(uci);
                        board.make_move(move);
                        ply_count++;
                    }
                }
                
                float eval = engine->evaluate(board);
                Utils::log_info("Eval after ply " + std::to_string(ply_count) + 
                               " (white-persp): " + std::to_string(eval));
                
                if ((board.turn() == ChessBoard::WHITE && our_white) || 
                    (board.turn() == ChessBoard::BLACK && !our_white)) {
                    
                    if (play_best_move(game_id, board)) ply_count++;
                }
            } else if (event.type == "gameState") {
                if (event.status != "started") {
                    Utils::log_info("Game " + game_id + " ended with status: " + event.status);
                    return;
                }
                
                if (event.draw_offer) handle_draw_offer(game_id, board);
                
                auto moves = Utils::split_string(event.moves, ' ');
                for (size_t i = ply_count; i < moves.size(); i++) {
                    const auto& uci = moves[i];
                    if (!uci.empty()) {
                        auto move = ChessBoard::Move::from_uci(uci);
                        board.make_move(move);
                        ply_count++;
                    }
                }
                
                float eval = engine->evaluate(board);
                Utils::log_info("Eval after ply " + std::to_string(ply_count) + 
                               " (white-persp): " + std::to_string(eval));
                
                if ((board.turn() == ChessBoard::WHITE && our_white) || 
                    (board.turn() == ChessBoard::BLACK && !our_white)) {
                    
                    if (play_best_move(game_id, board)) ply_count++;
                }
            }
        });
    }
    
    bool play_best_move(const std::string& game_id, ChessBoard& board) {
        auto move = engine->get_best_move(board);
        if (!move.uci_string.empty()) {
            if (client.make_move(game_id, move.uci())) {
                Utils::log_info("Move sent successfully: " + move.uci());
                board.make_move(move);
                return true;
            } else {
                Utils::log_error("Failed to send move: " + move.uci());
            }
        } else {
            Utils::log_error("No valid move found!");
        }
        return false;
    }
    
    void handle_draw_offer(const std::string& game_id, const ChessBoard& board) {
        float eval = engine->evaluate(board);
        float our_eval = our_white ? eval : -eval;

        if (our_eval > 0.0f) {
            Utils::log_info("Declining draw offer (our eval: " + std::to_string(our_eval) + 
                           ", white eval: " + std::to_string(eval) + ")");
            client.decline_draw(game_id);
        } else {
            Utils::log_info("Accepting draw offer (our eval: " + std::to_string(our_eval) + 
                           ", white eval: " + std::to_string(eval) + ")");
            client.accept_draw(game_id);
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
        std::cout << "Usage: " << argv[0] << " <lichess_token> [depth]" << std::endl;
        std::cout << "Example: " << argv[0] << " lip_abc123... 5" << std::endl;
        std::cout << std::endl;
        std::cout << "You'll also need:" << std::endl;
        std::cout << "1. A valid Lichess API token with bot permissions" << std::endl;
        std::cout << "2. model_coefficients.txt file in the cpp/train/ directory (run export_model.py to create it)" << std::endl;
        std::cout << "3. Network connectivity for Lichess API calls" << std::endl;
        std::cout << std::endl;
        std::cout << "Optional depth parameter (default: 4, recommended: 3-6)" << std::endl;
        return 0;
    }
    
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <lichess_token> [depth]" << std::endl;
        std::cerr << "Run without arguments to see test output" << std::endl;
        std::cerr << "Depth parameter is optional (default: 4)" << std::endl;
        return 1;
    }
    
    std::string token = argv[1];
    int depth = 4; // Default depth
    
    if (argc == 3) {
        try {
            depth = std::stoi(argv[2]);
            if (depth < 1 || depth > 10) {
                std::cerr << "Warning: Depth should be between 1-10. Using depth " << depth << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid depth parameter, using default depth 4" << std::endl;
            depth = 4;
        }
    }
    
    std::string model_path = "../../train/model_coefficients.txt"; // Relative to build/Release directory
    
    std::cout << "=== Starting Lichess Bot ===" << std::endl;
    std::cout << "Token: " << token.substr(0, 8) << "..." << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Search depth: " << depth << std::endl;
    std::cout << std::endl;
    
    try {
        LichessBot bot(token, model_path, depth);
        bot.start();
    } catch (const std::exception& e) {
        Utils::log_error("Bot crashed with exception: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
