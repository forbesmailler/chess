"""Continuously challenge online Lichess bots at various time controls."""

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def get_my_username():
    status, body = api_request("/account")
    if status == 200:
        return json.loads(body)["id"]
    sys.exit(f"Failed to get account: {body}")


def get_online_bots(nb=50):
    url = f"{BASE}/bot/online?nb={nb}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            bots = []
            for line in resp.read().decode().strip().split("\n"):
                if line.strip():
                    bots.append(json.loads(line))
            return bots
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
    print(f"  Challenge failed ({status}): {body[:200]}")
    return None


def cancel_challenge(challenge_id):
    api_request(f"/challenge/{challenge_id}/cancel", method="POST")


def main():
    my_id = get_my_username()
    print(f"Bot account: {my_id}")
    print(f"Time controls: {', '.join(f'{t // 60}+{i}' for t, i in TIME_CONTROLS)}")
    print()

    challenged = set()  # track recently challenged bots to avoid spamming
    games_played = 0
    tracked_games = set()  # game IDs we know about

    while True:
        ongoing = get_ongoing_games()
        ongoing_ids = {g["gameId"] for g in ongoing}

        # Detect finished games
        finished = tracked_games - ongoing_ids
        for gid in finished:
            games_played += 1
            print(f"Game {gid} finished. Total games played: {games_played}")
        tracked_games -= finished

        # At capacity — wait and check again
        if len(ongoing) >= MAX_CONCURRENT:
            time.sleep(GAME_POLL_INTERVAL)
            continue

        # Get online bots
        bots = get_online_bots()
        candidates = [b for b in bots if b["id"] != my_id and b["id"] not in challenged]

        if not candidates:
            if not ongoing:
                print("No new bots available, resetting challenge history...")
                challenged.clear()
            time.sleep(30 if not ongoing else GAME_POLL_INTERVAL)
            continue

        # Pick a random bot and time control
        bot = random.choice(candidates)
        bot_name = bot["username"]
        challenged.add(bot["id"])
        clock_limit, clock_increment = random.choice(TIME_CONTROLS)
        tc_str = f"{clock_limit // 60}+{clock_increment}"

        print(f"Challenging {bot_name} ({tc_str})...")
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
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
