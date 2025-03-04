from game import ChessGame
from cnn_bot import ChessBot

if __name__ == "__main__":
    game = ChessGame()
    bot = ChessBot(game, "pth/chess_model.pth")
    game.play(bot)