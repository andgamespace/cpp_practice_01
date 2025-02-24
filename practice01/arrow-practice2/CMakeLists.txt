cmake_minimum_required(VERSION 3.20) # Ensure compatibility with modern CMake features
project(arrow_practice2)

set(CMAKE_CXX_STANDARD 20)

#-----------------------------------------------------------------------------
# Vcpkg Integration: Ensure vcpkg toolchain file is used properly
#-----------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "/Users/anshc/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

#-----------------------------------------------------------------------------
# Add Executable
#-----------------------------------------------------------------------------
add_executable(arrow_practice2 main.cpp)

#-----------------------------------------------------------------------------
# Find Dependencies: Using vcpkg packages
#-----------------------------------------------------------------------------
# NOTE: These are suggested based on your `vcpkg list` output. Adjust or remove libraries based on what you actually need.

# Includes fmt, taskflow, arrow, and other libraries
find_package(fmt REQUIRED)                             # Formatting library
find_package(taskflow CONFIG REQUIRED)                 # Fast parallel tasking library
find_package(arrow CONFIG REQUIRED)                    # Apache Arrow
find_package(vincentlaucsb-csv-parser CONFIG REQUIRED) # Another CSV library
find_package(Cereal CONFIG REQUIRED)                   # Header-only serialization library
find_package(mlpack CONFIG REQUIRED)                   # Machine learning library
find_package(armadillo CONFIG REQUIRED)                # Linear algebra library
find_package(ensmallen CONFIG REQUIRED)                # Optimization library
find_package(absl CONFIG REQUIRED)                     # Abseil libraries
find_package(stb REQUIRED CONFIG)                      # Graphics and image utilities

#-----------------------------------------------------------------------------
# Include Directories
#-----------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------
# Link Libraries
#-----------------------------------------------------------------------------
# Link all necessary vcpkg libraries to your executable
target_link_libraries(arrow_practice2 PRIVATE
        fmt::fmt
        taskflow::taskflow
        Arrow::arrow
        vincentlaucsb-csv-parser
        Cereal
        mlpack
        armadillo
        Ensmallen
        absl::absl_strings
        stb::stb
)

#-----------------------------------------------------------------------------
# Optional: Set the runtime output directory
#-----------------------------------------------------------------------------
set_target_properties(arrow_practice2 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)