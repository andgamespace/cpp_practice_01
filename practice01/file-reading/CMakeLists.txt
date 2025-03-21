cmake_minimum_required(VERSION 3.30)
project(file_reading)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_GENERATOR "Ninja")

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using Vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Find required packages
find_package(Protobuf REQUIRED)
find_package(arrow CONFIG REQUIRED)
find_package(unofficial-vincentlaucsb-csv-parser CONFIG REQUIRED)
find_package(drogon CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Taskflow CONFIG REQUIRED)

# Try to find nlohmann_json, but don't require it
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, using bundled version")
    # Define a header-only library target for the bundled version
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/include)
    # Download the single header file if it doesn't exist
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/nlohmann/json.hpp")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/include/nlohmann")
        file(DOWNLOAD
            "https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp"
            "${CMAKE_SOURCE_DIR}/include/nlohmann/json.hpp"
            SHOW_PROGRESS
        )
    endif()
else()
    message(STATUS "Found nlohmann_json: ${nlohmann_json_DIR}")
endif()

# Optional PyTorch support
option(WITH_PYTORCH "Build with PyTorch support" OFF)
if(WITH_PYTORCH)
    find_package(Torch)
    if(Torch_FOUND)
        message(STATUS "PyTorch found, enabling PyTorch support")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        include_directories(${TORCH_INCLUDE_DIRS})
        add_definitions(-DPYTORCH_AVAILABLE=1)
    else()
        message(STATUS "PyTorch not found, disabling PyTorch support")
        add_definitions(-DPYTORCH_AVAILABLE=0)
    endif()
else()
    message(STATUS "PyTorch support disabled")
    add_definitions(-DPYTORCH_AVAILABLE=0)
endif()

# Build the executable (your server/backtest app)
add_executable(file_reading
        src/main.cpp
        src/DataLoader.cpp
        src/DataLoader.h
        src/FrontendController.cpp
        src/FrontendController.h
        src/MyWebSocketController.cpp
        src/MyWebSocketController.h
        src/BacktestEngine.cpp
        src/BacktestEngine.h
)

# Link libraries for the executable
if (DEFINED ARROW_BUILD_STATIC AND ARROW_BUILD_STATIC)
    target_link_libraries(file_reading PRIVATE Arrow::arrow_static)
else()
    target_link_libraries(file_reading PRIVATE Arrow::arrow_shared)
endif()

target_link_libraries(file_reading PRIVATE
    unofficial::vincentlaucsb-csv-parser::csv
    Drogon::Drogon
    pybind11::module
    spdlog::spdlog
    Taskflow::Taskflow
)

# Link nlohmann_json based on how it was found
if(nlohmann_json_FOUND)
    target_link_libraries(file_reading PRIVATE nlohmann_json::nlohmann_json)
else()
    target_link_libraries(file_reading PRIVATE nlohmann_json)
endif()

if(Torch_FOUND)
    target_link_libraries(file_reading PRIVATE ${TORCH_LIBRARIES})
endif()

# Build the Python module (shared library)
add_library(my_module SHARED
        src/BacktestEngine.cpp
        src/PyBindings.cpp
        src/DataLoader.cpp
        src/FrontendController.cpp
        src/MyWebSocketController.cpp
)

# Link libraries for the Python module
if (DEFINED ARROW_BUILD_STATIC AND ARROW_BUILD_STATIC)
    target_link_libraries(my_module PRIVATE Arrow::arrow_static)
else()
    target_link_libraries(my_module PRIVATE Arrow::arrow_shared)
endif()

target_link_libraries(my_module PRIVATE
    unofficial::vincentlaucsb-csv-parser::csv
    spdlog::spdlog
    Taskflow::Taskflow
    Drogon::Drogon
    pybind11::module
)

# Link nlohmann_json based on how it was found
if(nlohmann_json_FOUND)
    target_link_libraries(my_module PRIVATE nlohmann_json::nlohmann_json)
else()
    target_link_libraries(my_module PRIVATE nlohmann_json)
endif()

if(Torch_FOUND)
    target_link_libraries(my_module PRIVATE ${TORCH_LIBRARIES})
endif()

# Set properties for the Python module
set_target_properties(my_module PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    POSITION_INDEPENDENT_CODE ON
)

# Add install targets
install(TARGETS file_reading DESTINATION bin)
install(TARGETS my_module DESTINATION lib)
