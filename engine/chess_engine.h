#pragma once
#include "chess_board.h"
#include "logistic_model.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <atomic>

struct TranspositionEntry {
    float score;
    int depth;
    enum NodeType { EXACT, LOWER_BOUND, UPPER_BOUND } type;
    ChessBoard::Move best_move;
};

struct TimeControl {
    int time_left_ms;      // Time left in milliseconds
    int increment_ms;      // Increment per move in milliseconds
    int moves_to_go;       // Moves until time control (0 = no time control)
    
    TimeControl(int time_ms = 0, int inc_ms = 0, int mtg = 0) 
        : time_left_ms(time_ms), increment_ms(inc_ms), moves_to_go(mtg) {}
};

struct SearchResult {
    ChessBoard::Move best_move;
    float score;
    int depth_reached;
    std::chrono::milliseconds time_used;
    int nodes_searched;
};

class ChessEngine {
public:
    explicit ChessEngine(std::shared_ptr<LogisticModel> model, int max_time_ms = 1000);
    
    float evaluate(const ChessBoard& board);
    SearchResult get_best_move(const ChessBoard& board, const TimeControl& time_control);
    
    void set_max_time(int max_time_ms) { max_search_time_ms = max_time_ms; }
    int get_max_time() const { return max_search_time_ms; }
    void stop_search() { should_stop.store(true); }
    
private:
    static constexpr float MATE_VALUE = 10000.0f;
    static constexpr float SEARCH_INTERRUPTED = -99999.0f;  // Special value for interrupted search
    static constexpr size_t CACHE_SIZE = 1000000;
    
    std::shared_ptr<LogisticModel> model;
    int max_search_time_ms;
    mutable std::unordered_map<std::string, TranspositionEntry> transposition_table;
    mutable std::unordered_map<std::string, float> eval_cache;
    mutable std::atomic<bool> should_stop{false};
    mutable std::atomic<int> nodes_searched{0};
    
    // Time management
    int calculate_search_time(const TimeControl& time_control);
    
    // Search functions
    SearchResult iterative_deepening_search(const ChessBoard& board, int max_time_ms);
    float negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv = false);
    float quiescence_search(const ChessBoard& board, float alpha, float beta, int depth = 0);
    std::vector<ChessBoard::Move> order_moves(const ChessBoard& board, const std::vector<ChessBoard::Move>& moves, const ChessBoard::Move& tt_move = ChessBoard::Move{});
    int score_move(const ChessBoard& board, const ChessBoard::Move& move);
    void clear_cache_if_needed();
    std::string get_position_key(const ChessBoard& board) const;
};
