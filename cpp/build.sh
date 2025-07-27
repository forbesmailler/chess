#!/bin/bash

echo "Building Lichess Bot C++ Version"
echo "================================"

# Check if we're in the cpp directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Make sure you're in the cpp directory."
    exit 1
fi

# Export the model first
echo "Step 1: Exporting model coefficients..."
python3 export_model.py
if [ $? -ne 0 ]; then
    echo "Warning: Model export failed. The bot will use a dummy model."
    echo "Make sure you have the required Python packages (joblib, numpy)."
    echo "Continuing with build..."
fi

# Create build directory
echo "Step 2: Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Step 3: Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    echo "Make sure you have CMake installed and the required dependencies."
    echo "Dependencies: libcurl, nlohmann-json"
    exit 1
fi

# Build the project
echo "Step 4: Building the project..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed."
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "To test the chess implementation:"
echo "  ./test_chess"
echo ""
echo "To run the bot:"
echo "  ./lichess_bot YOUR_LICHESS_API_TOKEN"
echo ""
