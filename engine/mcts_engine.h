#pragma once

#include <random>
#include <vector>

#include "base_engine.h"

class MCTSEngine : public BaseEngine {
   public:
    explicit MCTSEngine(int max_time_ms = 30000,
                        EvalMode eval_mode = EvalMode::HANDCRAFTED,
                        std::shared_ptr<NNUEModel> nnue_model = nullptr);

    SearchResult get_best_move(const ChessBoard& board,
                               const TimeControl& time_control) override;
    float evaluate(const ChessBoard& board) override;

   private:
    struct MCTSNode {
        ChessBoard board;
        ChessBoard::Move move;
        MCTSNode* parent = nullptr;
        std::vector<std::unique_ptr<MCTSNode>> children;

        int visits = 0;
        float total_score = 0.0f;
        bool is_terminal = false;
        bool is_expanded = false;
        float prior_probability = 0.0f;

        mutable std::vector<ChessBoard::Move> legal_moves_cache;
        mutable bool legal_moves_cached = false;

        float get_average_score() const {
            return visits > 0 ? total_score / visits : 0.0f;
        }

        float get_uct_value(float exploration_constant, float log_parent,
                            float sqrt_parent) const;
        const std::vector<ChessBoard::Move>& get_legal_moves() const;
        bool is_leaf() const { return children.empty(); }
    };

    MCTSNode* select(MCTSNode* root);
    void expand(MCTSNode* node);
    float simulate(const ChessBoard& board);
    void backpropagate(MCTSNode* node, float score);

    float evaluate_position(const ChessBoard& board);

    float exploration_constant = config::mcts::EXPLORATION_CONSTANT;
    int max_simulation_depth = config::mcts::MAX_SIMULATION_DEPTH;

    mutable std::mt19937 rng;

    struct CacheEntry {
        uint64_t key = 0;
        float score = 0;
    };
    static constexpr size_t EVAL_CACHE_SIZE = config::search::EVAL_CACHE_SIZE;
    static constexpr size_t EVAL_CACHE_MASK = config::search::EVAL_CACHE_MASK;
    std::vector<CacheEntry> eval_cache;
};
