#pragma once
#include <chess.hpp>
#include <string>
#include <vector>

// Wrapper around the chess library to match our interface
class ChessBoard {
   public:
    enum Color { WHITE = 0, BLACK = 1 };

    // PieceType enum to match chess engine expectations
    enum PieceType { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NONE = 6 };

    struct Move {
        std::string uci_string;
        chess::Move internal_move;

        std::string uci() const { return uci_string; }

        static Move from_uci(const std::string& uci) {
            Move move;
            move.uci_string = uci;
            return move;
        }

        bool is_promotion() const;
        int from() const;
        int to() const;
    };

    struct CastlingRights {
        bool white_kingside = true;
        bool white_queenside = true;
        bool black_kingside = true;
        bool black_queenside = true;
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

    // Additional methods needed by chess engine
    PieceType piece_type_at(int square) const;
    PieceType piece_type_at(const std::string& square_str) const;
    bool is_capture_move(const Move& move) const;
    PieceType piece_at(int square) const;  // Alias for piece_type_at

    // Static utility methods
    static int square_from_string(const std::string& sq);
    static std::string square_to_string(int square);

   private:
    std::vector<chess::Move> move_history;
};
