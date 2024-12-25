from chess_game import ChessGame
from chess_bot import ChessBot

if __name__ == "__main__":
    game = ChessGame()
    bot = ChessBot(game)
    game.play(bot)