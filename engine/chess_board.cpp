#include "chess_board.h"

ChessBoard::ChessBoard() : board() {}

ChessBoard::ChessBoard(const std::string& fen) { load_fen(fen); }

void ChessBoard::load_fen(const std::string& fen) { board = chess::Board(fen); }

std::string ChessBoard::to_fen() const { return board.getFen(); }

std::vector<ChessBoard::Move> ChessBoard::get_legal_moves() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    std::vector<Move> result;
    result.reserve(moves.size());
    for (const auto& move : moves) {
        Move m;
        m.internal_move = move;
        result.push_back(m);
    }

    return result;
}

bool ChessBoard::make_move(const Move& move) {
    chess::Move chess_move = move.internal_move;
    if (chess_move == chess::Move::NO_MOVE) {
        try {
            chess_move = chess::uci::uciToMove(board, move.uci_string);
            if (chess_move == chess::Move::NO_MOVE) return false;
        } catch (...) {
            return false;
        }
    }
    move_history.push_back(chess_move);
    board.makeMove(chess_move);
    return true;
}

void ChessBoard::unmake_move(const Move& move) {
    if (!move_history.empty()) {
        board.unmakeMove(move_history.back());
        move_history.pop_back();
    }
}

bool ChessBoard::is_checkmate() const {
    auto [reason, result] = board.isGameOver();
    return reason == chess::GameResultReason::CHECKMATE;
}

bool ChessBoard::is_stalemate() const {
    auto [reason, result] = board.isGameOver();
    return reason == chess::GameResultReason::STALEMATE;
}

bool ChessBoard::is_draw() const {
    auto [reason, result] = board.isGameOver();
    return result == chess::GameResult::DRAW;
}

bool ChessBoard::is_game_over() const {
    auto [reason, result] = board.isGameOver();
    return result != chess::GameResult::NONE;
}

bool ChessBoard::is_in_check(Color color) const {
    return ((color == WHITE && board.sideToMove() == chess::Color::WHITE) ||
            (color == BLACK && board.sideToMove() == chess::Color::BLACK)) &&
           board.inCheck();
}

ChessBoard::Color ChessBoard::turn() const {
    return board.sideToMove() == chess::Color::WHITE ? WHITE : BLACK;
}

ChessBoard::CastlingRights ChessBoard::get_castling_rights() const {
    auto cr = board.castlingRights();
    using Side = chess::Board::CastlingRights::Side;
    return {
        cr.has(chess::Color::WHITE, Side::KING_SIDE),
        cr.has(chess::Color::WHITE, Side::QUEEN_SIDE),
        cr.has(chess::Color::BLACK, Side::KING_SIDE),
        cr.has(chess::Color::BLACK, Side::QUEEN_SIDE),
    };
}

int ChessBoard::piece_count() const { return board.occ().count(); }

int ChessBoard::square_from_string(const std::string& sq) {
    if (sq.length() != 2) return -1;
    int file = sq[0] - 'a';
    int rank = sq[1] - '1';
    return (file < 0 || file > 7 || rank < 0 || rank > 7) ? -1 : rank * 8 + file;
}

std::string ChessBoard::square_to_string(int square) {
    if (square < 0 || square > 63) return "";
    return std::string(1, 'a' + (square % 8)) + std::string(1, '1' + (square / 8));
}

ChessBoard::PieceType ChessBoard::piece_type_at(int square) const {
    if (square < 0 || square > 63) return NONE;

    chess::Square sq = static_cast<chess::Square>(square);
    chess::Piece piece = board.at(sq);
    if (piece == chess::Piece::NONE) return NONE;

    chess::PieceType pt = piece.type();
    static const PieceType piece_map[] = {PAWN,  KNIGHT, BISHOP, ROOK,
                                          QUEEN, KING,   NONE};
    return piece_map[static_cast<int>(pt)];
}

ChessBoard::PieceType ChessBoard::piece_type_at(const std::string& square_str) const {
    return piece_type_at(square_from_string(square_str));
}

bool ChessBoard::Move::is_promotion() const {
    return internal_move.typeOf() == chess::Move::PROMOTION;
}

int ChessBoard::Move::from() const { return internal_move.from().index(); }

int ChessBoard::Move::to() const { return internal_move.to().index(); }

bool ChessBoard::is_capture_move(const Move& move) const {
    return move.internal_move.typeOf() == chess::Move::ENPASSANT ||
           board.at(move.internal_move.to()) != chess::Piece::NONE;
}

ChessBoard::PieceType ChessBoard::piece_at(int square) const {
    return piece_type_at(square);
}
