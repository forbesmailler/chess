#pragma once

#include <string>
#include <vector>

#include "chess_board.h"

class NNUEModel {
   public:
    NNUEModel() = default;

    // Load binary weight file exported by export_nnue.py
    bool load_weights(const std::string& path);

    // Evaluate position, returns score from white's perspective in centipawns
    float predict(const ChessBoard& board) const;

    bool is_loaded() const { return loaded; }

   private:
    // 768 piece-square + 4 castling + 1 en passant
    static constexpr int INPUT_SIZE = 773;
    static constexpr int HIDDEN1_SIZE = 256;
    static constexpr int HIDDEN2_SIZE = 32;
    static constexpr int OUTPUT_SIZE = 3;
    static constexpr float MATE_VALUE = 10000.0f;

    // Weights and biases
    std::vector<float> w1;  // INPUT_SIZE x HIDDEN1_SIZE
    std::vector<float> b1;  // HIDDEN1_SIZE
    std::vector<float> w2;  // HIDDEN1_SIZE x HIDDEN2_SIZE
    std::vector<float> b2;  // HIDDEN2_SIZE
    std::vector<float> w3;  // HIDDEN2_SIZE x OUTPUT_SIZE
    std::vector<float> b3;  // OUTPUT_SIZE

    bool loaded = false;

    // Extract piece-square, castling, and en passant features (STM perspective)
    static std::vector<float> extract_features(const ChessBoard& board);

    static float clipped_relu(float x);
};
