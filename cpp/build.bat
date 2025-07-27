@echo off
echo Building Lichess Bot C++ Version
echo ================================

REM Check if we're in the cpp directory
if not exist "CMakeLists.txt" (
    echo Error: CMakeLists.txt not found. Make sure you're in the cpp directory.
    pause
    exit /b 1
)

REM Export the model first
echo Step 1: Exporting model coefficients...
python export_model.py
if %errorlevel% neq 0 (
    echo Warning: Model export failed. The bot will use a dummy model.
    echo Make sure you have the required Python packages (joblib, numpy).
    echo Continuing with build...
)

REM Create build directory
echo Step 2: Creating build directory...
if not exist "build" mkdir build
cd build

REM Configure with CMake (try vcpkg toolchain if available)
echo Step 3: Configuring with CMake...
set VCPKG_ROOT=C:\vcpkg
if exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo Using vcpkg toolchain...
    cmake .. -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
) else (
    echo Using system libraries...
    cmake ..
)

if %errorlevel% neq 0 (
    echo Error: CMake configuration failed.
    echo Make sure you have CMake installed and the required dependencies.
    echo Dependencies: libcurl, nlohmann-json
    pause
    exit /b 1
)

REM Build the project
echo Step 4: Building the project...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo Error: Build failed.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo.
echo To test the chess implementation:
echo   test_chess.exe
echo.
echo To run the bot:
echo   lichess_bot.exe YOUR_LICHESS_API_TOKEN
echo.
pause
