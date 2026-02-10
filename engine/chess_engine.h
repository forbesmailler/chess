#pragma once
#include <vector>

#include "base_engine.h"

struct EvalCacheEntry {
    uint64_t key = 0;
    float score = 0;
};

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
    static constexpr int TIME_CHECK_INTERVAL = config::search::TIME_CHECK_INTERVAL;
    static_assert((TIME_CHECK_INTERVAL & (TIME_CHECK_INTERVAL - 1)) == 0,
                  "TIME_CHECK_INTERVAL must be a power of 2");

    std::chrono::steady_clock::time_point search_deadline;
    void check_time();

    // TT: power-of-2 flat array for O(1) cache-friendly lookup
    static constexpr size_t TT_SIZE = config::search::TT_SIZE;
    static constexpr size_t TT_MASK = config::search::TT_MASK;
    std::vector<TranspositionEntry> transposition_table;

    static constexpr size_t EVAL_CACHE_SIZE = config::search::EVAL_CACHE_SIZE;
    static constexpr size_t EVAL_CACHE_MASK = config::search::EVAL_CACHE_MASK;
    std::vector<EvalCacheEntry> eval_cache;

    // Killer move heuristic: 2 killer moves per ply (chess::Move = 4 bytes)
    static constexpr int MAX_PLY = config::search::MAX_DEPTH + 10;
    chess::Move killers[MAX_PLY][2];

    // History heuristic: indexed by [from_square][to_square]
    int history[64][64];

    // Countermove heuristic: indexed by [prev_from][prev_to]
    chess::Move countermoves[64][64];

    static constexpr float ASPIRATION_DELTA = config::search::ASPIRATION_DELTA;

    SearchResult iterative_deepening_search(ChessBoard board, int max_time_ms);
    float negamax(ChessBoard& board, int depth, int ply, float alpha, float beta,
                  bool is_pv = false, chess::Move prev_move = chess::Move::NO_MOVE);
    float quiescence_search(ChessBoard& board, float alpha, float beta, int depth = 0,
                            bool in_check = false);
    float search_evaluate(const ChessBoard& board);
    void order_moves(const ChessBoard& board, chess::Movelist& moves,
                     chess::Move tt_move, int ply,
                     chess::Move prev_move = chess::Move::NO_MOVE);
    void score_moves(const ChessBoard& board, const chess::Movelist& moves, int* scores,
                     chess::Move tt_move, int ply,
                     chess::Move prev_move = chess::Move::NO_MOVE);
};
