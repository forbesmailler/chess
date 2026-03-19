# Deployment Guide

Deploy the chess bot to a Linux VPS as a systemd service.

## Prerequisites

- Linux VPS (Ubuntu/Debian recommended)
- CMake 3.16+, GCC/Clang with C++17 support
- libcurl-dev, nlohmann-json-dev
- A Lichess API token with bot permissions

## Build on the server

```bash
sudo apt install build-essential cmake libcurl4-openssl-dev nlohmann-json3-dev

cd ~/repos
git clone <your-repo-url> chess
cd chess/engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Copy NNUE weights to repo root
cp /path/to/nnue.bin ~/repos/chess/nnue.bin
```

## Install the systemd service

```bash
# Create the environment file with your token
echo 'LICHESS_TOKEN=lip_xxxxx' > ~/repos/chess/.env
chmod 600 ~/repos/chess/.env

# Install and start the service
sudo cp deploy/chess-bot.service /etc/systemd/system/chess-bot.service
sudo systemctl daemon-reload
sudo systemctl enable chess-bot
sudo systemctl start chess-bot

# Check status and logs
sudo systemctl status chess-bot
sudo journalctl -u chess-bot -f
```

## Update the bot

Manually:

```bash
cd ~/repos/chess
git pull
cd engine/build
cmake --build .
sudo systemctl restart chess-bot
```

Or via invoke (runs build, tests, install, and restart):

```bash
cd ~/repos/chess
invoke deploy
```
