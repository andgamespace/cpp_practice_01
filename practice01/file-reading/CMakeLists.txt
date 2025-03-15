cmake_minimum_required(VERSION 3.30)
project(file_reading)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_GENERATOR "Ninja")

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using Vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Ensure Protobuf is found (needed for Arrow::arrow_static)
find_package(Protobuf REQUIRED)

find_package(arrow CONFIG REQUIRED)
find_package(unofficial-vincentlaucsb-csv-parser CONFIG REQUIRED)
find_package(drogon CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Taskflow CONFIG REQUIRED)

# Set source files
add_executable(file_reading src/main.cpp
        src/DataLoader.cpp
        src/DataLoader.h
        src/FrontendController.cpp
        src/FrontendController.h
        src/MyWebSocketController.cpp
        src/MyWebSocketController.h
        src/BacktestEngine.cpp
        src/BacktestEngine.h
)

# Choose static or shared Arrow
if (DEFINED ARROW_BUILD_STATIC AND ARROW_BUILD_STATIC)
    target_link_libraries(file_reading PRIVATE Arrow::arrow_static)
else()
    target_link_libraries(file_reading PRIVATE Arrow::arrow_shared)
endif()

# Other libraries...
target_link_libraries(file_reading PRIVATE unofficial::vincentlaucsb-csv-parser::csv Drogon::Drogon pybind11::module spdlog::spdlog Taskflow::Taskflow)
