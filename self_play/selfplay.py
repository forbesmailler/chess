# selfplay_train.py
# Integrated self-play and online training with logging

import os
import random
import logging
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import chess
from network import ChessNet
from mcts import MCTS, state_to_tensor, UCI_TO_IDX, ALL_UCIS

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
GAMES_PER_EPOCH = 50
MCTS_SIMS = 100


def tensor_from_pi(pi_dict):
    """
    Convert a dict of move->prob into a tensor of length len(ALL_UCIS).
    """
    pi = torch.zeros(len(ALL_UCIS), dtype=torch.float32)
    for uci, p in pi_dict.items():
        pi[UCI_TO_IDX[uci]] = p
    return pi


def train_on_batch(model, optimizer, batch):
    """
    Perform a single optimization step on the given batch.
    Batch is a list of (fen, pi_dict, z).
    """
    # Prepare tensors
    states = torch.stack([state_to_tensor(chess.Board(fen)) for fen, pi, z in batch]).to(DEVICE)
    pis = torch.stack([tensor_from_pi(pi) for fen, pi, z in batch]).to(DEVICE)
    zs = torch.tensor([z for fen, pi, z in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    logits, values = model(states)
    loss_v = nn.MSELoss()(values, zs)
    loss_p = nn.CrossEntropyLoss()(logits, pis.argmax(dim=1))
    loss = loss_v + loss_p
    loss.backward()
    optimizer.step()
    return loss.item()


def selfplay_train_loop():
    """
    Main loop: for each epoch, generate games by self-play,
    add examples to replay buffer, and perform online training.
    """
    # Initialize model and optimizer
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Replay buffer
    buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS} start")
        for game_num in range(1, GAMES_PER_EPOCH + 1):
            # Self-play one game
            board = chess.Board()
            mcts = MCTS(model, sims=MCTS_SIMS, device=DEVICE)
            game_examples = []

            while not board.is_game_over():
                root = mcts.search(board)
                # Extract visit counts -> move probabilities
                counts = {mv.uci(): child.N for mv, child in root.children.items()}
                total = sum(counts.values())
                pi = {uci: n / total for uci, n in counts.items()}
                game_examples.append((board.fen(), pi, None))
                # Select best move
                best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
                board.push(best_move)

            # Determine game outcome
            result = board.result()  # '1-0', '0-1', or '1/2-1/2'
            z = 1.0 if result == '1-0' else (-1.0 if result == '0-1' else 0.0)
            game_data = [(fen, pi, z) for fen, pi, _ in game_examples]

            # Add to replay buffer
            buffer.extend(game_data)
            logger.info(
                f"Game {game_num}/{GAMES_PER_EPOCH}: generated {len(game_data)} examples, "
                f"buffer size={len(buffer)}"
            )

            # Online training step if enough data
            if len(buffer) >= BATCH_SIZE:
                batch = random.sample(buffer, BATCH_SIZE)
                loss = train_on_batch(model, optimizer, batch)
                logger.info(f"  Training step loss={loss:.4f}")

        # Save checkpoint at end of epoch
        ckpt_path = f'checkpoint_epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    best_path = 'best.pth'
    torch.save(model.state_dict(), best_path)
    logger.info(f"Training complete. Model saved to {best_path}")


if __name__ == '__main__':
    selfplay_train_loop()
