# C++ Lichess Bot

This is a C++ port of your Python Lichess bot, designed for improved performance while preserving all functionality.

## Features

- **Complete Chess Engine**: Uses the chess-library for full chess move generation and validation
- **Feature Extraction**: Matches the Python version's feature extraction exactly
- **Logistic Regression Model**: Loads and uses your trained scikit-learn model
- **Lichess API Integration**: Full integration with Lichess API for bot functionality
- **Caching**: Implements position evaluation caching for better performance
- **Logging**: Comprehensive logging system

## Dependencies

### Required Libraries

1. **libcurl** - For HTTP requests to Lichess API
2. **nlohmann/json** - For JSON parsing
3. **chess-library** - For complete chess move generation and validation (automatically downloaded by CMake)
4. **CMake** - For building the project

### Installing Dependencies

#### Windows (using vcpkg)
```powershell
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install dependencies
.\vcpkg install curl nlohmann-json
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install cmake libcurl4-openssl-dev nlohmann-json3-dev
```

#### macOS (using Homebrew)
```bash
brew install cmake curl nlohmann-json
```

## Building

1. **Export your trained model** (required first step):
   ```bash
   cd cpp
   python export_model.py
   ```
   This will create `model_coefficients.txt` from your `chess_lr.joblib` file.

2. **Create build directory and compile**:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

3. **On Windows with vcpkg**:
   ```powershell
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"
   cmake --build . --config Release
   ```

## Usage

1. **Run the bot**:
   ```bash
   ./lichess_bot <your_lichess_api_token>
   ```

   On Windows:
   ```powershell
   .\lichess_bot.exe <your_lichess_api_token>
   ```

2. **The bot will**:
   - Connect to Lichess using your API token
   - Accept challenges automatically
   - Play games using the loaded ML model
   - Log all activities to console

## File Structure

- **main.cpp** - Main bot application and game handling logic
- **chess_board.h/cpp** - Chess board representation and move generation
- **chess_engine.h/cpp** - Chess engine with negamax search
- **feature_extractor.h/cpp** - Feature extraction (matches Python version)
- **logistic_model.h/cpp** - Logistic regression model loader/predictor
- **lichess_client.h/cpp** - Lichess API client
- **utils.h/cpp** - Utility functions and logging
- **export_model.py** - Python script to export model coefficients

## Model Export Process

The C++ version cannot directly read scikit-learn's .joblib files, so you need to export the model coefficients:

1. Run the export script:
   ```bash
   python export_model.py
   ```

2. This creates `model_coefficients.txt` with the format:
   ```
   INTERCEPT
   <intercept_values>
   
   COEFFICIENTS
   <coefficient_values>
   ```

3. The C++ code automatically looks for this file when loading the model.

## Performance Improvements

The C++ version provides several performance benefits over the Python version:

1. **Faster Execution**: Native compiled code vs interpreted Python
2. **Better Memory Management**: More efficient memory usage
3. **Optimized Search**: Compiled negamax with efficient pruning
4. **Reduced Overhead**: No Python interpreter overhead
5. **Parallel Compilation**: Can be optimized with compiler flags

## Configuration

You can modify performance settings in the source code:

- **Search Depth**: Change `DEFAULT_DEPTH` in `chess_engine.h`
- **Cache Size**: Modify `CACHE_SIZE` in `chess_engine.h`
- **Model Path**: Update model path in `main.cpp`

## Troubleshooting

### Common Issues

1. **Model not loading**: Make sure you've run `export_model.py` first
2. **Build errors**: Check that all dependencies are installed
3. **API errors**: Verify your Lichess API token is valid and has bot permissions
4. **Missing moves**: The chess move generation is simplified - you may need to integrate a full chess library like `libchess` for complete functionality

### Incomplete Chess Implementation

**Note**: The chess implementation now uses the `chess-library` which provides complete chess functionality including:

✅ **Full move generation** - All legal moves including castling, en passant, promotion
✅ **Game state validation** - Proper check, checkmate, and stalemate detection  
✅ **Move making/unmaking** - Complete and correct position updates
✅ **FEN parsing** - Full FEN string support

The bot should now work correctly for production use!

### Extending the Bot

The code is modular and can be extended with:

- Opening book integration
- Endgame tablebase support
- Time management
- Pondering (thinking on opponent's time)
- Multiple model support

## License

This C++ port maintains the same functionality as your original Python bot while providing the performance benefits of compiled code.
