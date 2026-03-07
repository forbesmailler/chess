"""Play rated games against humans on Lichess.

Launches the bot executable with --training-data, then watches the TV
feed for the target speed category to discover active players near our
rating and challenges them directly.
"""

import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
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

MAX_CONCURRENT = load_engine()["bot"]["max_concurrent_games"]
GAME_POLL_INTERVAL = 10
CHALLENGE_TIMEOUT = 30
DEFAULT_RATING_RANGE = 200
# How many candidate players to collect before starting to challenge
MIN_CANDIDATES = 5
# Max candidates to keep in the pool
MAX_CANDIDATES = 200
# How long a candidate stays valid (seconds)
CANDIDATE_TTL = 600


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
    total = clock_limit + clock_increment * 40
    if total < 179:
        return "bullet"
    if total < 479:
        return "blitz"
    if total < 1499:
        return "rapid"
    return "classical"


def get_my_account():
    status, body = api_request("/account")
    if status == 200:
        return json.loads(body)
    sys.exit(f"Failed to get account: {body}")


def get_ongoing_games():
    status, body = api_request("/account/playing")
    if status == 200:
        return json.loads(body).get("nowPlaying", [])
    return []


def cancel_challenge(challenge_id):
    api_request(f"/challenge/{challenge_id}/cancel", method="POST")


def challenge_player(username, clock_limit, clock_increment):
    """Challenge a specific player. Returns challenge ID or None."""
    data = f"rated=true&clock.limit={clock_limit}&clock.increment={clock_increment}"
    status, body = api_request(f"/challenge/{username}", method="POST", data=data)
    if status == 200:
        return json.loads(body).get("challenge", {}).get("id")
    if status == 429:
        print("  Rate limited (429), backing off 60s...")
        time.sleep(60)
        return None
    print(f"  Challenge to {username} failed ({status}): {body[:200]}")
    return None


class PlayerPool:
    """Collects active player usernames from the Lichess TV feed."""

    def __init__(self, speed, my_id, my_rating, rating_range):
        self.speed = speed
        self.my_id = my_id
        self.my_rating = my_rating
        self.rating_range = rating_range
        self.lock = threading.Lock()
        # Map username -> (rating, last_seen_time)
        self._players = {}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._watch_tv, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def update_rating(self, new_rating):
        with self.lock:
            self.my_rating = new_rating

    def get_candidates(self, exclude):
        """Return list of (username, rating) within range, excluding given set."""
        now = time.time()
        with self.lock:
            # Expire old entries
            expired = [
                u for u, (_, t) in self._players.items() if now - t > CANDIDATE_TTL
            ]
            for u in expired:
                del self._players[u]

            results = []
            for username, (rating, _) in self._players.items():
                if username in exclude:
                    continue
                if abs(rating - self.my_rating) <= self.rating_range:
                    results.append((username, rating))
        return results

    @property
    def pool_size(self):
        with self.lock:
            return len(self._players)

    def _watch_tv(self):
        """Stream the TV feed for our speed category to discover active players."""
        url = f"{BASE}/tv/{self.speed}"
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/x-ndjson")
                req.add_header("Authorization", f"Bearer {TOKEN}")
                with urllib.request.urlopen(req, timeout=600) as resp:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            self._process_line(line.decode())
            except Exception as e:
                if not self._stop.is_set():
                    print(f"  TV feed error: {e}, reconnecting in 10s...")
                    time.sleep(10)

    def _process_line(self, line):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        players = data.get("players", {})
        now = time.time()
        with self.lock:
            for color in ("white", "black"):
                player = players.get(color, {})
                user = player.get("user", {})
                username = user.get("id", "")
                title = user.get("title", "")
                rating = player.get("rating", 0)

                # Skip bots, ourselves, and missing data
                if not username or username == self.my_id or title == "BOT":
                    continue

                self._players[username] = (rating, now)

                # Cap pool size by removing oldest entries
                if len(self._players) > MAX_CANDIDATES:
                    oldest = min(self._players, key=lambda u: self._players[u][1])
                    del self._players[oldest]


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--time", type=int, default=60, help="Clock time in seconds (default: 60)"
    )
    p.add_argument(
        "--increment",
        type=int,
        default=0,
        help="Clock increment in seconds (default: 0)",
    )
    p.add_argument(
        "--rating-range",
        type=int,
        default=DEFAULT_RATING_RANGE,
        help=f"Max rating difference (default: {DEFAULT_RATING_RANGE})",
    )
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

    # Launch bot subprocess
    dep = load_deploy()
    bot_exe = dep["paths"]["bot_exe"]
    bot_cmd = [bot_exe]
    if not args.no_training_data:
        bot_cmd += ["--training-data", args.training_data]
    print(f"Starting bot: {' '.join(bot_cmd)}")
    bot_proc = subprocess.Popen(bot_cmd, stdout=sys.stdout, stderr=sys.stderr)
    time.sleep(3)  # let the bot connect to Lichess

    if bot_proc.poll() is not None:
        sys.exit(f"Bot exited immediately with code {bot_proc.returncode}")

    account = get_my_account()
    my_id = account["id"]
    speed = tc_to_speed(args.time, args.increment)
    my_rating = account.get("perfs", {}).get(speed, {}).get("rating", 1500)
    tc_str = f"{args.time // 60}+{args.increment}"

    print(f"Bot account: {my_id}")
    print(f"Rating: {my_rating} {speed}")
    print(
        f"Mode: challenge active {speed} players within ±{args.rating_range} at {tc_str}"
    )
    print(f"Max concurrent: {MAX_CONCURRENT}")
    if not args.no_training_data:
        print(f"Training data: {args.training_data}")
    print()

    # Start watching TV feed to discover active players
    pool = PlayerPool(speed, my_id, my_rating, args.rating_range)
    pool.start()
    print(f"Watching {speed} TV feed for active players...")

    # Wait for some candidates before starting
    for _ in range(30):
        if pool.pool_size >= MIN_CANDIDATES:
            break
        time.sleep(1)
    print(f"Player pool: {pool.pool_size} players discovered")

    games_played = 0
    tracked_games = set()
    challenged = set()  # usernames we've already challenged (avoid spamming)

    try:
        while True:
            ongoing = get_ongoing_games()
            ongoing_ids = {g["gameId"] for g in ongoing}

            # Detect finished games
            finished = tracked_games - ongoing_ids
            for gid in finished:
                games_played += 1
                print(f"Game {gid} finished. Total: {games_played}")
            tracked_games -= finished

            # Refresh rating every 10 games
            if games_played > 0 and games_played % 10 == 0:
                account = get_my_account()
                new_rating = account.get("perfs", {}).get(speed, {}).get("rating", 1500)
                if new_rating != my_rating:
                    my_rating = new_rating
                    pool.update_rating(my_rating)
                    print(f"Rating updated: {my_rating}")

            # At capacity — wait
            if len(ongoing) >= MAX_CONCURRENT:
                time.sleep(GAME_POLL_INTERVAL)
                continue

            # Get candidates from the pool
            candidates = pool.get_candidates(exclude=challenged | {my_id})

            if not candidates:
                if not ongoing and challenged:
                    print(
                        f"No new candidates, resetting challenge history "
                        f"(pool: {pool.pool_size})..."
                    )
                    challenged.clear()
                time.sleep(GAME_POLL_INTERVAL)
                continue

            # Pick a random candidate
            username, rating = random.choice(candidates)
            challenged.add(username)

            print(f"Challenging {username} ({rating} {speed}) at {tc_str}...")
            challenge_id = challenge_player(username, args.time, args.increment)
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
                print(f"  {username} did not accept, cancelling.")
                cancel_challenge(challenge_id)
                time.sleep(2)
                continue

            # Game started
            print(f"  Game started: https://lichess.org/{challenge_id}")
            tracked_games.add(challenge_id)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pool.stop()
        bot_proc.terminate()
        try:
            bot_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bot_proc.kill()
        print("Bot stopped.")


if __name__ == "__main__":
    main()
