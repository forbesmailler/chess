#pragma once

#include "chess_board.h"

// Handcrafted evaluation function using tapered eval (middlegame/endgame blend).
// Returns score in centipawns from white's perspective.
float handcrafted_evaluate(const ChessBoard& board);
