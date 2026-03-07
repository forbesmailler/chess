"""Continuously challenge online Lichess bots at various time controls."""

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.load_config import deploy as load_deploy
from config.load_config import engine as load_engine

# Load .env
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

TOKEN = os.environ.get("LICHESS_TOKEN")
if not TOKEN:
    sys.exit("LICHESS_TOKEN not set")

BASE = "https://lichess.org/api"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}

# Time controls to rotate through: (clock_limit_seconds, increment_seconds)
TIME_CONTROLS = [
    (60, 0),  # 1+0 bullet
    (60, 1),  # 1+1 bullet
    (120, 1),  # 2+1 bullet
    (180, 0),  # 3+0 blitz
    (180, 2),  # 3+2 blitz
    (300, 0),  # 5+0 blitz
    (300, 3),  # 5+3 blitz
    (600, 0),  # 10+0 rapid
    (900, 10),  # 15+10 rapid
    (1800, 0),  # 30+0 classical
    (1800, 20),  # 30+20 classical
]

CHALLENGE_TIMEOUT = 30  # seconds to wait for challenge acceptance
GAME_POLL_INTERVAL = 10  # seconds between game status checks
MAX_GAME_WAIT = 3600  # max seconds to wait for a single game
MAX_CONCURRENT = load_engine()["bot"]["max_concurrent_games"]
DAILY_BOT_LIMIT = 95  # stay under Lichess's 100 bot-vs-bot games per rolling 24h window
DAILY_WINDOW = 24 * 3600  # 24 hours in seconds


def api_request(path, method="GET", data=None):
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers=HEADERS, method=method)
    if data:
        req.data = data.encode()
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        print(f"  Request error: {e}")
        return 0, ""


def tc_to_speed(clock_limit, clock_increment):
    """Map a time control to a Lichess speed category."""
    total = clock_limit + clock_increment * 40
    if total < 179:
        return "bullet"
    if total < 479:
        return "blitz"
    if total < 1499:
        return "rapid"
    return "classical"


def get_my_account():
    """Return full account data including ratings."""
    status, body = api_request("/account")
    if status == 200:
        return json.loads(body)
    sys.exit(f"Failed to get account: {body}")


def get_my_rating(account, speed):
    """Get our rating for a given speed category."""
    return account.get("perfs", {}).get(speed, {}).get("rating", 1500)


def get_online_bots(nb=1000):
    url = f"{BASE}/bot/online?nb={nb}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            bots = []
            for line in resp.read().decode().strip().split("\n"):
                if line.strip():
                    bots.append(json.loads(line))
            return bots
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  Rate limited (429) fetching bots, backing off 60s...")
            time.sleep(60)
        else:
            print(f"  Failed to fetch bots ({e.code}): {e.read().decode()[:200]}")
        return []
    except Exception as e:
        print(f"  Failed to fetch bots: {e}")
        return []


def get_ongoing_games():
    status, body = api_request("/account/playing")
    if status == 200:
        return json.loads(body).get("nowPlaying", [])
    return []


def wait_for_game_finish(game_id):
    """Poll until the game is no longer in our ongoing games list."""
    start = time.time()
    while time.time() - start < MAX_GAME_WAIT:
        games = get_ongoing_games()
        if not any(g["gameId"] == game_id for g in games):
            return True
        time.sleep(GAME_POLL_INTERVAL)
    return False


def challenge_bot(username, clock_limit, clock_increment):
    data = f"rated=true&clock.limit={clock_limit}&clock.increment={clock_increment}"
    status, body = api_request(f"/challenge/{username}", method="POST", data=data)
    if status == 200:
        return json.loads(body).get("challenge", {}).get("id")
    if status == 429:
        print("  Rate limited (429), backing off 60s...")
        time.sleep(60)
        return None
    print(f"  Challenge failed ({status}): {body[:200]}")
    return None


def cancel_challenge(challenge_id):
    api_request(f"/challenge/{challenge_id}/cancel", method="POST")


def launch_bot(training_data_path):
    """Launch the bot executable as a subprocess."""
    dep = load_deploy()
    bot_exe = dep["paths"]["bot_exe"]

    project_root = Path(__file__).resolve().parent.parent
    pointer_file = project_root / dep["paths"]["current_best_file"]
    nnue_weights = ""
    if pointer_file.exists():
        rel_path = pointer_file.read_text().strip()
        nnue_path = project_root / rel_path
        if nnue_path.exists():
            nnue_weights = str(nnue_path)

    bot_cmd = [bot_exe]
    if nnue_weights:
        bot_cmd += ["--eval=nnue", f"--nnue-weights={nnue_weights}"]
    book_path = project_root / "book.bin"
    if book_path.exists():
        bot_cmd.append(f"--book={book_path}")
    if training_data_path:
        bot_cmd.append(f"--training-data={training_data_path}")

    print(f"Starting bot: {' '.join(bot_cmd)}")
    proc = subprocess.Popen(bot_cmd, stdout=sys.stdout, stderr=sys.stderr)
    time.sleep(3)

    if proc.poll() is not None:
        sys.exit(f"Bot exited immediately with code {proc.returncode}")
    return proc


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--training-data",
        type=str,
        default="training_data.bin",
        help="Training data output file (default: training_data.bin)",
    )
    p.add_argument(
        "--no-training-data", action="store_true", help="Don't collect training data"
    )
    args = p.parse_args()

    bot_proc = launch_bot("" if args.no_training_data else args.training_data)

    account = get_my_account()
    my_id = account["id"]
    print(f"Bot account: {my_id}")
    print(f"Time controls: {', '.join(f'{t // 60}+{i}' for t, i in TIME_CONTROLS)}")
    if not args.no_training_data:
        print(f"Training data: {args.training_data}")
    print()

    challenged = set()
    game_timestamps = deque()
    tracked_games = set()

    try:
        while True:
            now = time.time()

            # Expire timestamps outside the 24h window
            while game_timestamps and now - game_timestamps[0] >= DAILY_WINDOW:
                game_timestamps.popleft()

            ongoing = get_ongoing_games()
            ongoing_ids = {g["gameId"] for g in ongoing}

            # Detect finished games
            finished = tracked_games - ongoing_ids
            for gid in finished:
                print(
                    f"Game {gid} finished. "
                    f"Daily games: {len(game_timestamps)}/{DAILY_BOT_LIMIT}"
                )
            tracked_games -= finished

            # Daily limit reached — wait until oldest game expires from the window
            if len(game_timestamps) >= DAILY_BOT_LIMIT:
                wait_until = game_timestamps[0] + DAILY_WINDOW
                wait_secs = max(0, wait_until - time.time())
                wait_mins = wait_secs / 60
                print(
                    f"Daily limit ({DAILY_BOT_LIMIT}) reached. "
                    f"Waiting {wait_mins:.0f}m until window rolls over..."
                )
                time.sleep(min(wait_secs + 1, 300))
                continue

            # At capacity — wait and check again
            if len(ongoing) >= MAX_CONCURRENT:
                time.sleep(GAME_POLL_INTERVAL)
                continue

            # Pick time control first so we can filter by rating
            clock_limit, clock_increment = random.choice(TIME_CONTROLS)
            speed = tc_to_speed(clock_limit, clock_increment)
            my_rating = get_my_rating(account, speed)
            tc_str = f"{clock_limit // 60}+{clock_increment}"

            # Get online bots and filter by rating proximity
            bots = get_online_bots()
            candidates = []
            for b in bots:
                if b["id"] == my_id or b["id"] in challenged:
                    continue
                bot_rating = b.get("perfs", {}).get(speed, {}).get("rating", 1500)
                if abs(bot_rating - my_rating) <= 500:
                    candidates.append(b)

            if not candidates:
                if not ongoing:
                    print(
                        f"No bots within 500 of {my_rating} {speed}, resetting history..."
                    )
                    challenged.clear()
                time.sleep(30 if not ongoing else GAME_POLL_INTERVAL)
                continue

            # Pick a random bot from filtered candidates
            bot = random.choice(candidates)
            bot_name = bot["username"]
            bot_rating = bot.get("perfs", {}).get(speed, {}).get("rating", 1500)
            challenged.add(bot["id"])

            print(f"Challenging {bot_name} ({bot_rating} {speed}) at {tc_str}...")
            challenge_id = challenge_bot(bot["id"], clock_limit, clock_increment)
            if not challenge_id:
                time.sleep(2)
                continue

            # Wait for acceptance
            accepted = False
            for _ in range(CHALLENGE_TIMEOUT // 3):
                time.sleep(3)
                ongoing = get_ongoing_games()
                if any(g["gameId"] == challenge_id for g in ongoing):
                    accepted = True
                    break

            if not accepted:
                print(f"  {bot_name} did not accept, cancelling.")
                cancel_challenge(challenge_id)
                time.sleep(2)
                continue

            # Game started
            print(f"  Game started: https://lichess.org/{challenge_id}")
            tracked_games.add(challenge_id)
            game_timestamps.append(time.time())
            print(f"  Daily games: {len(game_timestamps)}/{DAILY_BOT_LIMIT}")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bot_proc.terminate()
        try:
            bot_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bot_proc.kill()
        print("Bot stopped.")


if __name__ == "__main__":
    main()
