# prepare_data.py
import csv

import chess
import chess.pgn
from tqdm import tqdm


def split_pgn_to_csv(pgn_path, train_csv, val_csv, flush_every=100):
    with (
        open(pgn_path, encoding="utf-8") as pgn,
        open(train_csv, "w", newline="", encoding="utf-8") as f_train,
        open(val_csv, "w", newline="", encoding="utf-8") as f_val,
    ):
        w_train = csv.writer(f_train)
        w_val = csv.writer(f_val)
        w_train.writerow(["FEN", "value"])
        w_val.writerow(["FEN", "value"])

        train_buf = []
        val_buf = []

        game_idx = 0
        pbar = tqdm(desc="Games", unit="game")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_idx += 1
            pbar.update(1)

            res = game.headers.get("Result", "")
            if res == "1-0":
                gval = 1.0
            elif res == "0-1":
                gval = -1.0
            elif res in ("1/2-1/2", "½-½"):
                gval = 0.0
            else:
                continue

            board = game.board()
            target_buf = val_buf if (game_idx % 10) == 0 else train_buf

            for move in game.mainline_moves():
                board.push(move)
                target_buf.append([board.fen(), gval])

            if game_idx % flush_every == 0:
                if train_buf:
                    w_train.writerows(train_buf)
                    train_buf.clear()
                if val_buf:
                    w_val.writerows(val_buf)
                    val_buf.clear()
                pbar.set_postfix(flushed_games=game_idx)

        if train_buf:
            w_train.writerows(train_buf)
        if val_buf:
            w_val.writerows(val_buf)

        pbar.close()
        print(f"Done. Processed {game_idx} games → {train_csv} & {val_csv}")


if __name__ == "__main__":
    split_pgn_to_csv(
        "C:/Users/forbe/Downloads/lichess_db_standard_rated_2014-07.pgn/lichess_db_standard_rated_2014-07.pgn",
        "train.csv",
        "val.csv",
    )
