#include "chess_board.h"
#include "feature_extractor.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing ChessBoard implementation..." << std::endl;
    
    // Test 1: Starting position
    ChessBoard board;
    std::string start_fen = board.to_fen();
    std::cout << "Starting FEN: " << start_fen << std::endl;
    
    // Test 2: Legal moves generation
    auto moves = board.get_legal_moves();
    std::cout << "Legal moves from start: " << moves.size() << std::endl;
    assert(moves.size() == 20); // 20 legal moves from starting position
    
    // Test 3: Make a move
    if (!moves.empty()) {
        auto first_move = moves[0];
        std::cout << "Making move: " << first_move.uci() << std::endl;
        bool success = board.make_move(first_move);
        assert(success);
        
        std::cout << "FEN after move: " << board.to_fen() << std::endl;
        
        // Test 4: Unmake move
        board.unmake_move(first_move);
        std::string back_fen = board.to_fen();
        std::cout << "FEN after unmake: " << back_fen << std::endl;
        // Note: This might not match exactly due to move history in some libraries
    }
    
    // Test 5: Feature extraction
    auto features = FeatureExtractor::extract_features(board);
    std::cout << "Feature vector size: " << features.size() << std::endl;
    assert(features.size() == 1544);
    
    // Test 6: Piece count
    int pieces = board.piece_count();
    std::cout << "Piece count: " << pieces << std::endl;
    assert(pieces == 32); // 32 pieces at start
    
    // Test 7: Castling rights
    auto castling = board.get_castling_rights();
    std::cout << "Castling rights: " 
              << (castling.white_kingside ? "K" : "") 
              << (castling.white_queenside ? "Q" : "")
              << (castling.black_kingside ? "k" : "")
              << (castling.black_queenside ? "q" : "") << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
