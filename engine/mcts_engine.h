#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "base_engine.h"
#include "chess_board.h"
#include "logistic_model.h"

class MCTSEngine : public BaseEngine {
   public:
    explicit MCTSEngine(std::shared_ptr<LogisticModel> model, int max_time_ms = 30000);
    ~MCTSEngine() = default;

    SearchResult get_best_move(const ChessBoard& board, const TimeControl& time_control) override;
    float evaluate(const ChessBoard& board) override;
    int get_max_time() const override { return max_search_time_ms; }

   private:
    struct MCTSNode {
        ChessBoard board;
        ChessBoard::Move move;  // Move that led to this position
        MCTSNode* parent = nullptr;
        std::vector<std::unique_ptr<MCTSNode>> children;

        // MCTS statistics
        int visits = 0;
        float total_score = 0.0f;
        bool is_terminal = false;
        bool is_expanded = false;
        float prior_probability = 0.0f;

        // Caching
        mutable std::vector<ChessBoard::Move> legal_moves_cache;
        mutable bool legal_moves_cached = false;

        float get_average_score() const { return visits > 0 ? total_score / visits : 0.0f; }

        float get_uct_value(float exploration_constant, int parent_visits) const;
        const std::vector<ChessBoard::Move>& get_legal_moves() const;
        bool is_leaf() const { return children.empty(); }
    };

    // Core MCTS methods
    MCTSNode* select(MCTSNode* root);
    void expand(MCTSNode* node);
    float simulate(const ChessBoard& board);
    void backpropagate(MCTSNode* node, float score);

    // Helper methods
    float evaluate_position(const ChessBoard& board);
    float get_move_prior(const ChessBoard& board, const ChessBoard::Move& move);
    int calculate_search_time(const TimeControl& time_control);

    // Configuration
    float exploration_constant = 1.4f;  // UCT exploration parameter
    int max_simulation_depth = 100;

    // Random number generation for simulations
    mutable std::mt19937 rng;

    // Position evaluation cache
    mutable std::unordered_map<std::string, float> eval_cache;
    mutable std::mutex eval_cache_mutex;

    // Constants
    static constexpr int CACHE_SIZE = 100000;

    std::string get_position_key(const ChessBoard& board) const;
};
