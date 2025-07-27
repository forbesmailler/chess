# C++ Lichess Bot

This is a high-performance C++ port of your Python Lichess bot, designed for improved speed while preserving all functionality.

## Quick Setup Guide

Follow these steps to get the bot running:

### 1. Install Dependencies

#### Windows (Recommended)
```powershell
# Install vcpkg package manager
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install required libraries
.\vcpkg install curl nlohmann-json
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cmake libcurl4-openssl-dev nlohmann-json3-dev
```

#### macOS
```bash
brew install cmake curl nlohmann-json
```

### 2. Export Your Trained Model

The C++ bot can't read Python's `.joblib` files directly, so export your model:

```bash
cd cpp
python export_model.py
```

This creates `model_coefficients.txt` in the cpp directory.

### 3. Build the Bot

#### Windows (with vcpkg)
```powershell
cd cpp
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
```

#### Linux/macOS
```bash
cd cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### 4. Run the Bot

```bash
# Test mode (verify everything works)
./lichess_bot

# Bot mode (connect to Lichess)
./lichess_bot <your_lichess_api_token>
```

On Windows:
```powershell
# Test mode
.\lichess_bot.exe

# Bot mode  
.\lichess_bot.exe <your_lichess_api_token>
```

## What You Need

1. **Lichess API Token**: Get one from [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token) with bot permissions
2. **Your Trained Model**: The `chess_lr.joblib` file from your Python training
3. **C++ Build Tools**: Visual Studio 2022 (Windows) or GCC/Clang (Linux/macOS)

## Features

✅ **Complete Functionality:**
- Full chess engine with negamax search
- 1544-dimensional feature extraction (identical to Python version)
- Trained ML model integration
- Lichess API connectivity
- Automatic challenge acceptance
- Real-time game playing

✅ **Performance Benefits:**
- Native compiled code (much faster than Python)
- Efficient memory management
- Optimized search algorithms
- Reduced overhead

## Troubleshooting

**Model not loading?**
- Make sure you ran `python export_model.py` in the cpp directory
- Check that `model_coefficients.txt` exists in the cpp folder

**Build errors?**
- Verify all dependencies are installed
- On Windows, ensure you're using the vcpkg toolchain file

**API connection issues?**
- Verify your Lichess token is valid and has bot permissions
- Check your internet connection

## File Structure

- `main.cpp` - Bot application and game handling
- `chess_board.h/cpp` - Chess board using chess-library
- `chess_engine.h/cpp` - Negamax search engine  
- `feature_extractor.h/cpp` - Feature extraction (matches Python)
- `logistic_model.h/cpp` - ML model loader
- `lichess_client.h/cpp` - Lichess API integration
- `export_model.py` - Converts .joblib to text format
- `model_coefficients.txt` - Exported model data (created by export script)
