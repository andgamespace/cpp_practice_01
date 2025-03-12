cmake_minimum_required(VERSION 3.30)
project(file_reading)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Ninja build system
set(CMAKE_GENERATOR "Ninja")

# Include Vcpkg toolchain, adjust the VCPKG_ROOT if needed
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "Using Vcpkg toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# Locate dependencies via Vcpkg
find_package(arrow CONFIG REQUIRED)                                # Apache Arrow for CSV handling
find_package(unofficial-vincentlaucsb-csv-parser CONFIG REQUIRED)  # CSV Parser
find_package(drogon CONFIG REQUIRED)                               # Drogon HTTP framework
find_package(pybind11 CONFIG REQUIRED)                             # Pybind11
find_package(spdlog CONFIG REQUIRED)                               # spdlog logging library
find_package(Taskflow CONFIG REQUIRED)
# Set source files
add_executable(file_reading src/main.cpp
        src/DataLoader.cpp
        src/DataLoader.h
        src/FrontendController.cpp
        src/FrontendController.h)

# Apache Arrow library: Use the static or shared version of the library
if (DEFINED ARROW_BUILD_STATIC AND ARROW_BUILD_STATIC)
    target_link_libraries(file_reading PRIVATE Arrow::arrow_static)
else()
    target_link_libraries(file_reading PRIVATE Arrow::arrow_shared)
endif()

# vincentlaucsb-csv-parser: Link the provided target
target_link_libraries(file_reading PRIVATE unofficial::vincentlaucsb-csv-parser::csv)

# Drogon HTTP framework
target_link_libraries(file_reading PRIVATE Drogon::Drogon)

# Pybind11
target_link_libraries(file_reading PRIVATE pybind11::module)

# spdlog logging library
target_link_libraries(file_reading PRIVATE spdlog::spdlog)

# taskflow for concurrency
target_link_libraries(file_reading PRIVATE Taskflow::Taskflow)
