#include "chess_board.h"
#include "chess_engine.h"
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
    LichessBot(const std::string& token, const std::string& model_path)
        : client(token), model_path(model_path) {
        
        // Load the model
        model = std::make_shared<LogisticModel>();
        if (!model->load_model(model_path)) {
            Utils::log_warning("Failed to load model from " + model_path + ", using dummy model");
        }
        
        // Create the engine
        engine = std::make_unique<ChessEngine>(model);
        
        // Get account info
        if (!client.get_account_info(account_info)) {
            Utils::log_error("Failed to get account information");
        } else {
            Utils::log_info("Bot started as user: " + account_info.username + " (" + account_info.id + ")");
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
    
    void handle_event(const LichessClient::GameEvent& event) {
        if (event.type == "challenge") {
            Utils::log_info("Received challenge: " + event.challenge_id);
            if (client.accept_challenge(event.challenge_id)) {
                Utils::log_info("Accepted challenge: " + event.challenge_id);
            } else {
                Utils::log_error("Failed to accept challenge: " + event.challenge_id);
            }
        } else if (event.type == "gameStart") {
            Utils::log_info("Game started: " + event.game_id);
            std::thread game_thread(&LichessBot::handle_game, this, event.game_id);
            game_thread.detach();
        }
    }
    
    void handle_game(const std::string& game_id) {
        Utils::log_info("Handling game: " + game_id);
        
        bool first_event = true;
        bool our_white = false;
        ChessBoard board;
        int ply_count = 0;
        
        client.stream_game(game_id, [&](const LichessClient::GameEvent& event) {
            if (event.type == "gameFull" && first_event) {
                first_event = false;
                our_white = event.is_white;
                Utils::log_info("Game " + game_id + ": we are " + (our_white ? "White" : "Black"));
                
                // Parse initial moves
                auto moves = Utils::split_string(event.moves, ' ');
                for (const auto& uci : moves) {
                    if (!uci.empty()) {
                        auto move = ChessBoard::Move::from_uci(uci);
                        board.make_move(move);
                        ply_count++;
                    }
                }
                
                // Make initial move if it's our turn
                if ((board.turn() == ChessBoard::WHITE && our_white) || 
                    (board.turn() == ChessBoard::BLACK && !our_white)) {
                    
                    float eval = engine->evaluate(board);
                    Utils::log_info("Eval after ply " + std::to_string(ply_count) + 
                                   " (white-persp): " + std::to_string(eval));
                    
                    if (play_best_move(game_id, board)) {
                        ply_count++;
                    }
                }
            } else if (event.type == "gameState") {
                if (event.status != "started") {
                    Utils::log_info("Game " + game_id + " ended with status: " + event.status);
                    return;
                }
                
                // Parse new moves
                auto moves = Utils::split_string(event.moves, ' ');
                for (size_t i = ply_count; i < moves.size(); i++) {
                    const auto& uci = moves[i];
                    if (!uci.empty()) {
                        std::string actor = ((ply_count % 2 == 0 && our_white) || 
                                           (ply_count % 2 == 1 && !our_white)) ? "Bot" : "Opponent";
                        Utils::log_info("Game " + game_id + ": " + actor + " played move " + uci);
                        
                        auto move = ChessBoard::Move::from_uci(uci);
                        board.make_move(move);
                        ply_count++;
                    }
                }
                
                float eval = engine->evaluate(board);
                Utils::log_info("Eval after ply " + std::to_string(ply_count) + 
                               " (white-persp): " + std::to_string(eval));
                
                // Make our move if it's our turn
                if ((board.turn() == ChessBoard::WHITE && our_white) || 
                    (board.turn() == ChessBoard::BLACK && !our_white)) {
                    
                    if (play_best_move(game_id, board)) {
                        ply_count++;
                    }
                }
            }
        });
    }
    
    bool play_best_move(const std::string& game_id, ChessBoard& board) {
        auto move = engine->get_best_move(board);
        if (!move.uci_string.empty()) { // Valid move check
            if (client.make_move(game_id, move.uci())) {
                board.make_move(move);
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <lichess_token>" << std::endl;
        return 1;
    }
    
    std::string token = argv[1];
    std::string model_path = "../engine/chess_lr.joblib"; // Relative to cpp directory
    
    try {
        LichessBot bot(token, model_path);
        bot.start();
    } catch (const std::exception& e) {
        Utils::log_error("Bot crashed with exception: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
