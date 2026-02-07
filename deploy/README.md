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

cd /opt
sudo git clone <your-repo-url> chess-bot-src
cd chess-bot-src/engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

sudo mkdir -p /opt/chess-bot
sudo cp Release/lichess_bot /opt/chess-bot/
# Copy NNUE weights if using NNUE eval
sudo cp /path/to/nnue.bin /opt/chess-bot/
```

## Install the systemd service

```bash
# Edit the service file to set your Lichess token
sudo cp deploy/chess-bot.service /etc/systemd/system/chess-bot@.service

# Enable and start (replace TOKEN with your Lichess API token)
sudo systemctl daemon-reload
sudo systemctl enable chess-bot@TOKEN
sudo systemctl start chess-bot@TOKEN

# Check status and logs
sudo systemctl status chess-bot@TOKEN
sudo journalctl -u chess-bot@TOKEN -f
```

## Update the bot

```bash
cd /opt/chess-bot-src
git pull
cd engine/build
cmake --build . --config Release
sudo cp Release/lichess_bot /opt/chess-bot/
sudo systemctl restart chess-bot@TOKEN
```
