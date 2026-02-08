#include "mcts_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

MCTSEngine::MCTSEngine(int max_time_ms, EvalMode eval_mode,
                       std::shared_ptr<NNUEModel> nnue_model)
    : BaseEngine(max_time_ms, eval_mode, std::move(nnue_model)),
      rng(std::random_device{}()) {
    eval_cache.reserve(CACHE_SIZE);
}

SearchResult MCTSEngine::get_best_move(const ChessBoard& board,
                                       const TimeControl& time_control) {
    auto legal_moves = board.get_legal_moves();

    if (legal_moves.empty()) {
        float score = board.is_in_check(board.turn()) ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }
    if (legal_moves.size() == 1) {
        return {legal_moves[0], 0.0f, 1, std::chrono::milliseconds(50), 1};
    }

    int search_time = calculate_search_time(time_control);
    auto start_time = std::chrono::steady_clock::now();
    should_stop.store(false);
    nodes_searched.store(0);

    // Create root node
    auto root = std::make_unique<MCTSNode>();
    root->board = board;
    root->visits = 1;

    // MCTS main loop
    int iterations = 0;
    while (!should_stop.load()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= search_time) break;

        // MCTS iteration: Select -> Expand -> Simulate -> Backpropagate
        MCTSNode* leaf = select(root.get());

        if (leaf && !leaf->is_terminal) {
            expand(leaf);

            // If expansion created children, select one for simulation
            if (!leaf->children.empty()) {
                leaf = leaf->children[0].get();  // Select first child for simulation
            }
        }

        if (leaf) {
            float score = simulate(leaf->board);
            backpropagate(leaf, score);
        }

        iterations++;
        if (iterations % 1000 == 0) {
            nodes_searched.store(iterations);
        }
    }

    // Select best move based on visit count (most robust)
    ChessBoard::Move best_move;
    int max_visits = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& child : root->children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move;
            best_score = child->get_average_score();
        }
    }

    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);

    return {best_move, best_score, 0, time_used, iterations};
}

float MCTSEngine::evaluate(const ChessBoard& board) { return evaluate_position(board); }

MCTSEngine::MCTSNode* MCTSEngine::select(MCTSNode* root) {
    MCTSNode* current = root;

    // Traverse tree until we reach a leaf node
    while (!current->is_leaf() && !current->is_terminal) {
        MCTSNode* best_child = nullptr;
        float best_uct = -std::numeric_limits<float>::infinity();

        for (const auto& child : current->children) {
            float uct = child->get_uct_value(exploration_constant, current->visits);
            if (uct > best_uct) {
                best_uct = uct;
                best_child = child.get();
            }
        }

        if (best_child) {
            current = best_child;
        } else {
            break;
        }
    }

    return current;
}

void MCTSEngine::expand(MCTSNode* node) {
    if (node->is_expanded || node->is_terminal) return;

    const auto& legal_moves = node->get_legal_moves();
    if (legal_moves.empty()) {
        node->is_terminal = true;
        return;
    }

    // Evaluate parent once for all move priors
    float parent_eval = evaluate_position(node->board);

    // Create child nodes for all legal moves
    node->children.reserve(legal_moves.size());
    for (const auto& move : legal_moves) {
        auto child = std::make_unique<MCTSNode>();
        child->board = node->board;
        child->move = move;
        child->parent = node;

        // Apply the move
        if (child->board.make_move(move)) {
            // Set prior probability based on move evaluation
            child->prior_probability = get_move_prior(node->board, move, parent_eval);

            // Check if this is a terminal position
            auto child_legal_moves = child->board.get_legal_moves();
            if (child_legal_moves.empty()) {
                child->is_terminal = true;
            }

            node->children.push_back(std::move(child));
        }
    }

    node->is_expanded = true;
}

float MCTSEngine::simulate(const ChessBoard& board) {
    ChessBoard sim_board = board;
    int depth = 0;

    // Random simulation with early termination based on evaluation
    while (depth < max_simulation_depth) {
        auto legal_moves = sim_board.get_legal_moves();
        if (legal_moves.empty()) {
            // Game over
            if (sim_board.is_checkmate()) {
                return sim_board.turn() == board.turn() ? -MATE_VALUE : MATE_VALUE;
            } else {
                return 0.0f;  // Stalemate or draw
            }
        }

        // Use evaluation to guide simulation occasionally
        if (depth % config::mcts::EVAL_FREQUENCY == 0) {
            float eval = evaluate_position(sim_board);
            // Early termination if position is very good/bad
            if (std::abs(eval) > MATE_VALUE * config::mcts::EARLY_TERMINATION_FACTOR) {
                return sim_board.turn() == board.turn() ? -eval : eval;
            }
        }

        // Select random move (could be improved with better heuristics)
        std::uniform_int_distribution<> dist(0, legal_moves.size() - 1);
        ChessBoard::Move selected_move = legal_moves[dist(rng)];

        if (!sim_board.make_move(selected_move)) {
            break;
        }

        depth++;
    }

    // Evaluate final position
    float eval = evaluate_position(sim_board);
    return sim_board.turn() == board.turn() ? -eval : eval;
}

void MCTSEngine::backpropagate(MCTSNode* node, float score) {
    MCTSNode* current = node;
    bool flip_score = false;

    while (current != nullptr) {
        current->visits++;
        current->total_score += flip_score ? -score : score;

        current = current->parent;
        flip_score = !flip_score;  // Flip score for alternating players
    }
}

float MCTSEngine::evaluate_position(const ChessBoard& board) {
    uint64_t pos_key = board.hash();

    {
        std::shared_lock<std::shared_mutex> lock(eval_cache_mutex);
        if (auto it = eval_cache.find(pos_key); it != eval_cache.end()) {
            return it->second;
        }
    }

    float eval = raw_evaluate(board);

    {
        std::unique_lock<std::shared_mutex> lock(eval_cache_mutex);
        if (eval_cache.size() < CACHE_SIZE) {
            eval_cache.emplace(pos_key, eval);
        }
    }

    return eval;
}

float MCTSEngine::get_move_prior(const ChessBoard& board, const ChessBoard::Move& move,
                                 float parent_eval) {
    ChessBoard temp_board = board;
    if (!temp_board.make_move(move)) return 0.0f;

    float eval_after = evaluate_position(temp_board);

    // Normalize the improvement
    float improvement = board.turn() == ChessBoard::WHITE ? eval_after - parent_eval
                                                          : parent_eval - eval_after;

    // Convert to probability (sigmoid-like function)
    return 1.0f / (1.0f + std::exp(-improvement / config::mcts::PRIOR_SIGMOID_SCALE));
}

float MCTSEngine::MCTSNode::get_uct_value(float exploration_constant,
                                          int parent_visits) const {
    if (visits == 0) {
        return std::numeric_limits<float>::infinity();  // Unvisited nodes have highest
                                                        // priority
    }

    float exploitation = get_average_score();
    float exploration =
        exploration_constant * std::sqrt(std::log(parent_visits) / visits);

    // Add prior knowledge
    float prior_bonus = prior_probability * exploration_constant *
                        std::sqrt(parent_visits) / (1 + visits);

    return exploitation + exploration + prior_bonus;
}

const std::vector<ChessBoard::Move>& MCTSEngine::MCTSNode::get_legal_moves() const {
    if (!legal_moves_cached) {
        legal_moves_cache = board.get_legal_moves();
        legal_moves_cached = true;
    }
    return legal_moves_cache;
}
