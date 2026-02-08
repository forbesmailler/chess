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
# Create the environment file with your token
echo 'LICHESS_TOKEN=lip_xxxxx' | sudo tee /opt/chess-bot/.env
sudo chmod 600 /opt/chess-bot/.env

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

```bash
cd /opt/chess-bot-src
git pull
cd engine/build
cmake --build . --config Release
sudo cp Release/lichess_bot /opt/chess-bot/
sudo systemctl restart chess-bot
```
