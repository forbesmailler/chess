#include "feature_extractor.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#include "chess_board.h"

namespace {
int get_piece_type(char c) {
    const char* pieces = "pnbrqk";
    const char* pos = std::strchr(pieces, std::tolower(c));
    return pos ? static_cast<int>(pos - pieces + 1) : 0;
}
}  // namespace

std::vector<float> FeatureExtractor::extract_features(const std::string& fen) {
    return extract_features(ChessBoard(fen));
}

std::vector<float> FeatureExtractor::extract_features(const ChessBoard& board) {
    auto piece_features = extract_piece_features(board);
    auto additional_features = extract_additional_features(board);

    std::vector<float> base_features;
    base_features.reserve(770);
    base_features.insert(base_features.end(), piece_features.begin(), piece_features.end());
    base_features.insert(base_features.end(), additional_features.begin(),
                         additional_features.end());

    float factor =
        static_cast<float>(board.piece_count() - 2) / config::features::PIECE_COUNT_DIVISOR;

    std::vector<float> features;
    features.reserve(FEATURE_SIZE);

    // Add scaled features (770 * 2 = 1540)
    for (float f : base_features) features.push_back(f * factor);
    for (float f : base_features) features.push_back(f * (1.0f - factor));

    // Add mobility features
    auto mobility_features = extract_mobility_features(board);
    features.insert(features.end(), mobility_features.begin(), mobility_features.end());

    return features;
}

std::array<float, 768> FeatureExtractor::extract_piece_features(const ChessBoard& board) {
    std::array<float, 768> piece_arr;
    piece_arr.fill(0.0f);

    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str;
    iss >> board_str;

    int square = 56;

    for (char c : board_str) {
        if (c == '/') {
            square -= 16;
        } else if (std::isdigit(c)) {
            square += (c - '0');
        } else if (int piece_type = get_piece_type(c); piece_type > 0) {
            int idx = (piece_type - 1) + (std::isupper(c) ? 0 : 6);
            piece_arr[idx * 64 + square] = 1.0f;
            square++;
        }
    }

    return piece_arr;
}

std::array<float, 2> FeatureExtractor::extract_additional_features(const ChessBoard& board) {
    // Parse FEN components
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;

    bool current_in_check = board.is_in_check(board.turn());

    // Check opponent by flipping turn
    std::string flipped_turn = (turn_str == "w") ? "b" : "w";
    std::string flipped_fen = board_str + " " + flipped_turn + " " + castling_str + " " + ep_str +
                              " " + halfmove_str + " " + fullmove_str;

    bool opponent_in_check = false;
    ChessBoard flipped_board(flipped_fen);
    if (!flipped_board.is_game_over()) {
        ChessBoard::Color opponent_color =
            (board.turn() == ChessBoard::WHITE) ? ChessBoard::BLACK : ChessBoard::WHITE;
        opponent_in_check = flipped_board.is_in_check(opponent_color);
    }

    // Return features based on colors (always white first, then black)
    if (board.turn() == ChessBoard::WHITE) {
        return {current_in_check ? 1.0f : 0.0f, opponent_in_check ? 1.0f : 0.0f};
    } else {
        return {opponent_in_check ? 1.0f : 0.0f, current_in_check ? 1.0f : 0.0f};
    }
}

std::array<float, 2> FeatureExtractor::extract_mobility_features(const ChessBoard& board) {
    std::array<float, 2> mobility_features = {0.0f, 0.0f};

    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;

    int piece_counts[2] = {0, 0};
    for (char c : board_str) {
        if (c != '/' && !std::isdigit(c)) {
            piece_counts[std::islower(c) ? 1 : 0]++;
        }
    }

    ChessBoard::Color current_turn = board.turn();
    auto current_moves = board.get_legal_moves();
    const char* turn_chars = "wb";

    for (int color = 0; color < 2; ++color) {
        if (piece_counts[color] >= 8) continue;

        auto color_enum = color == 0 ? ChessBoard::WHITE : ChessBoard::BLACK;
        float moves = 0.0f;

        if (current_turn == color_enum) {
            moves = static_cast<float>(current_moves.size());
        } else {
            std::string flipped_fen = board_str + " " + turn_chars[color] + " " + castling_str +
                                      " " + ep_str + " " + halfmove_str + " " + fullmove_str;
            ChessBoard flipped_board(flipped_fen);
            if (!flipped_board.is_game_over()) {
                moves = static_cast<float>(flipped_board.get_legal_moves().size());
            }
        }

        float factor = std::max((8.0f - piece_counts[color]) / 6.0f, 0.0f);
        mobility_features[color] = factor * moves;
    }

    return mobility_features;
}
