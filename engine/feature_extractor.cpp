#include "feature_extractor.h"
#include "chess_board.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace {
    int get_piece_type(char c) {
        switch (std::tolower(c)) {
            case 'p': return 1;
            case 'n': return 2;
            case 'b': return 3;
            case 'r': return 4;
            case 'q': return 5;
            case 'k': return 6;
            default: return 0;
        }
    }
}

std::vector<float> FeatureExtractor::extract_features(const std::string& fen) {
    return extract_features(ChessBoard(fen));
}

std::vector<float> FeatureExtractor::extract_features(const ChessBoard& board) {
    auto piece_features = extract_piece_features(board);
    auto additional_features = extract_additional_features(board);
    
    std::vector<float> base_features;
    base_features.reserve(770);
    base_features.insert(base_features.end(), piece_features.begin(), piece_features.end());
    base_features.insert(base_features.end(), additional_features.begin(), additional_features.end());
    
    float factor = static_cast<float>(board.piece_count() - 2) / 30.0f;
    
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
    std::string flipped_fen = board_str + " " + flipped_turn + " " + castling_str + " " + 
                             ep_str + " " + halfmove_str + " " + fullmove_str;
    
    bool opponent_in_check = false;
    ChessBoard flipped_board(flipped_fen);
    if (!flipped_board.is_game_over()) {
        ChessBoard::Color opponent_color = (board.turn() == ChessBoard::WHITE) ? ChessBoard::BLACK : ChessBoard::WHITE;
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
    std::array<float, 2> mobility_features = {0.0f, 0.0f};  // [white_mobility, black_mobility]
    
    // Parse FEN once to count pieces and extract components
    std::string fen = board.to_fen();
    std::istringstream iss(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    iss >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Count pieces for each color
    int white_pieces = 0;
    int black_pieces = 0;
    for (char c : board_str) {
        if (c != '/' && !std::isdigit(c)) {
            if (std::isupper(c)) white_pieces++;
            else if (std::islower(c)) black_pieces++;
        }
    }
    
    ChessBoard::Color current_turn = board.turn();
    auto current_moves = board.get_legal_moves();
    
    // Calculate white mobility if needed
    if (white_pieces < 8) {
        float white_moves = 0.0f;
        
        if (current_turn == ChessBoard::WHITE) {
            white_moves = static_cast<float>(current_moves.size());
        } else {
            // Get white moves by flipping turn
            std::string white_fen = board_str + " w " + castling_str + " " + 
                                   ep_str + " " + halfmove_str + " " + fullmove_str;
            ChessBoard white_board(white_fen);
            if (!white_board.is_game_over()) {
                white_moves = static_cast<float>(white_board.get_legal_moves().size());
            }
        }
        
        float white_factor = std::max((8.0f - white_pieces) / 6.0f, 0.0f);
        mobility_features[0] = white_factor * white_moves;
    }
    
    // Calculate black mobility if needed
    if (black_pieces < 8) {
        float black_moves = 0.0f;
        
        if (current_turn == ChessBoard::BLACK) {
            black_moves = static_cast<float>(current_moves.size());
        } else {
            // Get black moves by flipping turn
            std::string black_fen = board_str + " b " + castling_str + " " + 
                                   ep_str + " " + halfmove_str + " " + fullmove_str;
            ChessBoard black_board(black_fen);
            if (!black_board.is_game_over()) {
                black_moves = static_cast<float>(black_board.get_legal_moves().size());
            }
        }
        
        float black_factor = std::max((8.0f - black_pieces) / 6.0f, 0.0f);
        mobility_features[1] = black_factor * black_moves;
    }
    
    return mobility_features;
}
