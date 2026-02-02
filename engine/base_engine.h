#pragma once

#include <atomic>
#include <chrono>
#include <memory>

#include "chess_board.h"
#include "logistic_model.h"

struct TimeControl {
    int time_left_ms;
    int increment_ms;
};

struct SearchResult {
    ChessBoard::Move best_move;
    float score;
    int depth;
    std::chrono::milliseconds time_used;
    int nodes_searched;
};

// Abstract base class for chess engines
class BaseEngine {
   public:
    explicit BaseEngine(std::shared_ptr<LogisticModel> model, int max_time_ms = 1000)
        : model(model), max_search_time_ms(max_time_ms) {}

    virtual ~BaseEngine() = default;

    // Pure virtual functions that must be implemented by derived engines
    virtual float evaluate(const ChessBoard& board) = 0;
    virtual SearchResult get_best_move(const ChessBoard& board,
                                       const TimeControl& time_control) = 0;

    // Common interface methods
    virtual void set_max_time(int max_time_ms) { max_search_time_ms = max_time_ms; }
    virtual int get_max_time() const { return max_search_time_ms; }
    virtual void stop_search() { should_stop.store(true); }

   protected:
    static constexpr float MATE_VALUE = 10000.0f;

    std::shared_ptr<LogisticModel> model;
    int max_search_time_ms;
    mutable std::atomic<bool> should_stop{false};
    mutable std::atomic<int> nodes_searched{0};
};
