#pragma once
#include <string>
#include <vector>
#include <chess.hpp>

// Wrapper around the chess library to match our interface
class ChessBoard {
public:
    enum Color { WHITE = 0, BLACK = 1 };

    struct Move {
        std::string uci_string;
        
        std::string uci() const { return uci_string; }
        static Move from_uci(const std::string& uci) {
            Move move;
            move.uci_string = uci;
            return move;
        }
    };

    struct CastlingRights {
        bool white_kingside;
        bool white_queenside;
        bool black_kingside;
        bool black_queenside;
        
        CastlingRights() : white_kingside(true), white_queenside(true), 
                          black_kingside(true), black_queenside(true) {}
    };

    ChessBoard();
    explicit ChessBoard(const std::string& fen);

    // Core chess functionality using the library
    chess::Board board;
    
    // Interface methods
    void load_fen(const std::string& fen);
    std::string to_fen() const;
    std::vector<Move> get_legal_moves() const;
    bool make_move(const Move& move);
    void unmake_move(const Move& move);
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_draw() const;
    bool is_game_over() const;
    bool is_in_check(Color color) const;
    
    // Accessors for compatibility
    Color turn() const;
    CastlingRights get_castling_rights() const;
    int piece_count() const;
    
    // Static utility methods
    static int square_from_string(const std::string& sq);
    static std::string square_to_string(int square);
    
private:
    std::vector<chess::Move> move_history;
};
