cmake_minimum_required(VERSION 3.16)
project(lichess_bot)

option(BUILD_TESTS "Build unit tests" ON)

# CMake 4.x compatibility - handle version detection
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    cmake_policy(SET CMP0000 NEW)
    if(POLICY CMP0167)
        cmake_policy(SET CMP0167 NEW)
    endif()
endif()

# Prefer extraction timestamps for FetchContent downloads
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Debug information
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(MSVC)
        # MSVC optimizations
        add_compile_options(/O2 /Ob2 /Oi /Ot /GL)
        add_link_options(/LTCG)
        message(STATUS "Added MSVC optimization flags")
    else()
        # GCC/Clang optimizations
        add_compile_options(-O3 -march=native -flto)
        add_link_options(-flto)
        message(STATUS "Added GCC/Clang optimization flags")
    endif()
endif()

# Include FetchContent module
include(FetchContent)

# Platform-specific configuration
if(WIN32)
    # Windows-specific configuration
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    
    # Find CURL package (installed via vcpkg)
    find_package(CURL REQUIRED)
    if(CURL_FOUND)
        message(STATUS "CURL found: ${CURL_VERSION_STRING}")
        message(STATUS "CURL target: CURL::libcurl")
        set(SKIP_HTTP_CLIENT FALSE)
    else()
        message(FATAL_ERROR "CURL not found. Install with: vcpkg install curl")
    endif()
    
    # Find nlohmann_json package
    find_package(nlohmann_json QUIET)
    
    # nlohmann_json is easier to build, so we can still fetch it
    if(NOT nlohmann_json_FOUND)
        message(STATUS "nlohmann_json not found via vcpkg, using FetchContent...")
        
        FetchContent_Declare(
            nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
            URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
        )
        
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install OFF CACHE INTERNAL "")
        
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
    
else()
    # Linux/macOS configuration
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CURL REQUIRED libcurl)
    find_package(nlohmann_json REQUIRED)
endif()

# Chess library (single header library)
FetchContent_Declare(
    chess-library
    GIT_REPOSITORY https://github.com/Disservin/chess-library.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

# Set additional options that might help with CMake 4.x
set(BUILD_TESTING ${BUILD_TESTS} CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Google Test for unit testing
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# Chess library is header-only with no CMakeLists.txt — download and create target
FetchContent_MakeAvailable(chess-library)
add_library(chess-library-iface INTERFACE)
target_include_directories(chess-library-iface INTERFACE ${chess-library_SOURCE_DIR}/include)

# Create library for engine code (used by both main executable and tests)
add_library(chess_engine_lib STATIC
    base_engine.cpp
    chess_board.cpp
    chess_engine.cpp
    mcts_engine.cpp
    handcrafted_eval.cpp
    nnue_model.cpp
    self_play.cpp
    utils.cpp
)

target_link_libraries(chess_engine_lib PUBLIC chess-library-iface)

# Threading support (for self-play)
find_package(Threads REQUIRED)
target_link_libraries(chess_engine_lib PUBLIC Threads::Threads)

# Create the main executable
add_executable(lichess_bot
    main.cpp
    lichess_client.cpp
)

# Link libraries to main executable
target_link_libraries(lichess_bot
    chess_engine_lib
)

# Handle nlohmann_json linking
if(nlohmann_json_FOUND)
    target_link_libraries(lichess_bot nlohmann_json::nlohmann_json)
else()
    target_link_libraries(lichess_bot nlohmann_json)
endif()

# Platform-specific linking
if(WIN32)
    if(CURL_FOUND)
        target_link_libraries(lichess_bot CURL::libcurl)
    endif()
    
    # Windows-specific libraries
    target_link_libraries(lichess_bot ws2_32 wldap32 crypt32 advapi32)
    
    # CURL is required on Windows, so no conditional compilation needed
    
else()
    # Linux/macOS linking
    target_link_libraries(lichess_bot ${CURL_LIBRARIES})
    target_include_directories(lichess_bot PRIVATE ${CURL_INCLUDE_DIRS})
    target_compile_options(lichess_bot PRIVATE ${CURL_CFLAGS_OTHER})
endif()

# Set output directory for Windows
if(WIN32)
    set_target_properties(lichess_bot PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
    )
endif()

# Unit tests
if(BUILD_TESTS)
    add_executable(chess_tests
        ../tests/engine/test_chess_board.cpp
        ../tests/engine/test_chess_engine.cpp
        ../tests/engine/test_handcrafted_eval.cpp
        ../tests/engine/test_self_play.cpp
        ../tests/engine/test_nnue_model.cpp
    )

    target_include_directories(chess_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(chess_tests
        chess_engine_lib
        GTest::gtest_main
    )

    # Handle nlohmann_json linking for tests
    if(nlohmann_json_FOUND)
        target_link_libraries(chess_tests nlohmann_json::nlohmann_json)
    else()
        target_link_libraries(chess_tests nlohmann_json)
    endif()

    include(GoogleTest)
    gtest_discover_tests(chess_tests)

    if(WIN32)
        set_target_properties(chess_tests PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        )
    endif()
endif()