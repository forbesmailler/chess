#include <gtest/gtest.h>

#include "../chess_board.h"

TEST(ChessBoardTest, DefaultConstructor) {
    ChessBoard board;
    EXPECT_EQ(board.to_fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

TEST(ChessBoardTest, FenConstructor) {
    std::string fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2";
    ChessBoard board(fen);
    EXPECT_EQ(board.to_fen(), fen);
}

TEST(ChessBoardTest, LegalMovesFromStart) {
    ChessBoard board;
    auto moves = board.get_legal_moves();
    EXPECT_EQ(moves.size(), 20);  // 16 pawn moves + 4 knight moves
}

TEST(ChessBoardTest, MakeAndUnmakeMove) {
    ChessBoard board;
    std::string initial_fen = board.to_fen();

    auto moves = board.get_legal_moves();
    ASSERT_FALSE(moves.empty());

    auto move = moves[0];
    board.make_move(move);
    EXPECT_NE(board.to_fen(), initial_fen);

    board.unmake_move(move);
    EXPECT_EQ(board.to_fen(), initial_fen);
}

TEST(ChessBoardTest, TurnAlternates) {
    ChessBoard board;
    EXPECT_EQ(board.turn(), ChessBoard::WHITE);

    auto moves = board.get_legal_moves();
    board.make_move(moves[0]);
    EXPECT_EQ(board.turn(), ChessBoard::BLACK);

    moves = board.get_legal_moves();
    board.make_move(moves[0]);
    EXPECT_EQ(board.turn(), ChessBoard::WHITE);
}

TEST(ChessBoardTest, CastlingRightsInitial) {
    ChessBoard board;
    auto rights = board.get_castling_rights();
    EXPECT_TRUE(rights.white_kingside);
    EXPECT_TRUE(rights.white_queenside);
    EXPECT_TRUE(rights.black_kingside);
    EXPECT_TRUE(rights.black_queenside);
}

TEST(ChessBoardTest, PieceCount) {
    ChessBoard board;
    EXPECT_EQ(board.piece_count(), 32);
}

TEST(ChessBoardTest, IsCheckmate) {
    // Fool's mate position
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    EXPECT_TRUE(board.is_checkmate());
    EXPECT_TRUE(board.is_game_over());
}

TEST(ChessBoardTest, IsStalemate) {
    // Stalemate position: black king on a8, white king on b6, white queen on c7
    ChessBoard board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
    EXPECT_TRUE(board.is_stalemate());
    EXPECT_TRUE(board.is_game_over());
}

TEST(ChessBoardTest, MoveFromUci) {
    ChessBoard board;
    auto move = ChessBoard::Move::from_uci("e2e4");
    EXPECT_EQ(move.uci(), "e2e4");

    EXPECT_TRUE(board.make_move(move));
    EXPECT_EQ(board.turn(), ChessBoard::BLACK);
}

TEST(ChessBoardTest, PieceTypeAt) {
    ChessBoard board;
    EXPECT_EQ(board.piece_type_at(0), ChessBoard::ROOK);    // a1
    EXPECT_EQ(board.piece_type_at(4), ChessBoard::KING);    // e1
    EXPECT_EQ(board.piece_type_at(8), ChessBoard::PAWN);    // a2
    EXPECT_EQ(board.piece_type_at(32), ChessBoard::NONE);   // empty square
}

TEST(ChessBoardTest, SquareConversion) {
    EXPECT_EQ(ChessBoard::square_from_string("a1"), 0);
    EXPECT_EQ(ChessBoard::square_from_string("h8"), 63);
    EXPECT_EQ(ChessBoard::square_from_string("e4"), 28);

    EXPECT_EQ(ChessBoard::square_to_string(0), "a1");
    EXPECT_EQ(ChessBoard::square_to_string(63), "h8");
    EXPECT_EQ(ChessBoard::square_to_string(28), "e4");
}
