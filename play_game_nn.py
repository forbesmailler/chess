from chess_game import ChessGame
from chess_bot_nn import ChessBot

if __name__ == "__main__":
    game = ChessGame()
    bot = ChessBot(game, "pkl/logistic_model.pkl", "pkl/logistic_scaler.pkl")
    game.play(bot)