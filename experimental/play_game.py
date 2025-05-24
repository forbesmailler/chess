from chess_game import ChessGame, ChessBot

if __name__ == "__main__":
    game = ChessGame()
    bot = ChessBot(game, "pth/chess_model.pth")
    game.play(bot)