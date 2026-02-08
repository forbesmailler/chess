#pragma once
#include <cstring>
#include <unordered_map>
#include <vector>

#include "base_engine.h"

struct TranspositionEntry {
    uint64_t key = 0;
    float score = 0;
    int depth = 0;
    enum NodeType { EXACT, LOWER_BOUND, UPPER_BOUND } type = EXACT;
    chess::Move best_move = chess::Move::NO_MOVE;
};

class ChessEngine : public BaseEngine {
   public:
    explicit ChessEngine(int max_time_ms = 1000,
                         EvalMode eval_mode = EvalMode::HANDCRAFTED,
                         std::shared_ptr<NNUEModel> nnue_model = nullptr);

    float evaluate(const ChessBoard& board) override;
    SearchResult get_best_move(const ChessBoard& board,
                               const TimeControl& time_control) override;
    void clear_caches();

   private:
    static constexpr float SEARCH_INTERRUPTED = -99999.0f;
    static constexpr size_t CACHE_SIZE = config::search::CACHE_SIZE;
    static constexpr int TIME_CHECK_INTERVAL = config::search::TIME_CHECK_INTERVAL;
    static_assert((TIME_CHECK_INTERVAL & (TIME_CHECK_INTERVAL - 1)) == 0,
                  "TIME_CHECK_INTERVAL must be a power of 2");

    std::chrono::steady_clock::time_point search_deadline;
    void check_time();

    // TT: power-of-2 flat array for O(1) cache-friendly lookup
    static constexpr size_t TT_SIZE = 1 << 20;  // ~1M entries
    static constexpr size_t TT_MASK = TT_SIZE - 1;
    std::vector<TranspositionEntry> transposition_table;

    mutable std::unordered_map<uint64_t, float> eval_cache;

    // Killer move heuristic: 2 killer moves per ply (indexed by remaining depth)
    static constexpr int MAX_PLY = config::search::MAX_DEPTH + 10;
    ChessBoard::Move killers[MAX_PLY][2];

    // History heuristic: indexed by [from_square][to_square]
    int history[64][64];

    // Aspiration window
    static constexpr float ASPIRATION_DELTA = 50.0f;

    SearchResult iterative_deepening_search(const ChessBoard& board, int max_time_ms);
    float negamax(const ChessBoard& board, int depth, int ply, float alpha, float beta,
                  bool is_pv = false);
    float quiescence_search(const ChessBoard& board, float alpha, float beta,
                            int depth = 0, bool in_check = false);
    void order_moves(const ChessBoard& board, std::vector<ChessBoard::Move>& moves,
                     const ChessBoard::Move& tt_move, int ply);
};
