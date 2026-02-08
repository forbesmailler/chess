#pragma once
#include <unordered_map>
#include <vector>

#include "base_engine.h"

struct TranspositionEntry {
    float score;
    int depth;
    enum NodeType { EXACT, LOWER_BOUND, UPPER_BOUND } type;
    ChessBoard::Move best_move;
};

class ChessEngine : public BaseEngine {
   public:
    explicit ChessEngine(int max_time_ms = 1000,
                         EvalMode eval_mode = EvalMode::HANDCRAFTED,
                         std::shared_ptr<NNUEModel> nnue_model = nullptr);

    float evaluate(const ChessBoard& board) override;
    SearchResult get_best_move(const ChessBoard& board,
                               const TimeControl& time_control) override;

   private:
    static constexpr float SEARCH_INTERRUPTED = -99999.0f;
    static constexpr size_t CACHE_SIZE = config::search::CACHE_SIZE;
    static constexpr int TIME_CHECK_INTERVAL = config::search::TIME_CHECK_INTERVAL;

    std::chrono::steady_clock::time_point search_deadline;
    void check_time();

    mutable std::unordered_map<uint64_t, TranspositionEntry> transposition_table;
    mutable std::unordered_map<uint64_t, float> eval_cache;

    SearchResult iterative_deepening_search(const ChessBoard& board, int max_time_ms);
    float negamax(const ChessBoard& board, int depth, float alpha, float beta,
                  bool is_pv = false);
    float quiescence_search(const ChessBoard& board, float alpha, float beta,
                            int depth = 0);
    std::vector<ChessBoard::Move> order_moves(
        const ChessBoard& board, const std::vector<ChessBoard::Move>& moves,
        const ChessBoard::Move& tt_move = ChessBoard::Move{});
};
