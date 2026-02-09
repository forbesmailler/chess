#pragma once

#include <istream>
#include <string>
#include <vector>

#include "chess_board.h"
#include "generated_config.h"

class NNUEModel {
   public:
    NNUEModel() = default;

    // Load binary weight file exported by export_nnue.py
    bool load_weights(const std::string& path);

    // Load weights from an already-open stream (for in-memory loading)
    bool load_weights(std::istream& stream);

    // Evaluate position, returns score from white's perspective in centipawns
    float predict(const ChessBoard& board) const;

    bool is_loaded() const { return loaded; }

   private:
    static constexpr int INPUT_SIZE = config::nnue::INPUT_SIZE;
    static constexpr int HIDDEN1_SIZE = config::nnue::HIDDEN1_SIZE;
    static constexpr int HIDDEN2_SIZE = config::nnue::HIDDEN2_SIZE;
    static constexpr int OUTPUT_SIZE = config::nnue::OUTPUT_SIZE;
    static constexpr float MATE_VALUE = config::MATE_VALUE;

    // Weights and biases
    std::vector<float>
        w1;  // INPUT_SIZE x HIDDEN1_SIZE (row-major, for sparse accumulation)
    std::vector<float> b1;  // HIDDEN1_SIZE
    std::vector<float>
        w2;  // HIDDEN2_SIZE x HIDDEN1_SIZE (transposed at load for sequential access)
    std::vector<float> b2;  // HIDDEN2_SIZE
    std::vector<float> w3;  // HIDDEN2_SIZE x OUTPUT_SIZE
    std::vector<float> b3;  // OUTPUT_SIZE

    bool loaded = false;

    // Collect active (non-zero) feature indices (all features are binary)
    static void extract_features(const ChessBoard& board, std::vector<int>& active);
};
