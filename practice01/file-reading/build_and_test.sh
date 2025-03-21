#!/bin/bash
set -e

echo "=== Trading Environment Build and Test Script ==="

# Create build directory if it doesn't exist
mkdir -p build

# Configure with CMake
echo "Configuring with CMake..."
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cd ..

# Create symbolic link to stock data
echo "Creating symbolic link to stock data..."
mkdir -p build/src
ln -sf $(pwd)/src/stock_data build/src/stock_data

# Build the project
echo "Building the project..."
cmake --build build -- -j$(nproc)

# Run C++ tests
echo "Running C++ tests..."
cd build
./file_reading --test
cd ..

# Run Python tests
echo "Running Python tests..."

# Check if we have a conda/mamba environment
if command -v mamba &> /dev/null; then
    echo "Checking for mamba environment..."
    echo "Mamba found!"
    echo "Activating mamba environment..."
    mamba shell.bash activate
elif command -v conda &> /dev/null; then
    echo "Checking for conda environment..."
    echo "Conda found!"
    echo "Activating conda environment..."
    conda activate
else
    echo "No conda/mamba environment found, using system Python"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run Python tests
echo "Running Python test script..."
python test.py

# Run technical indicators test
echo "Running technical indicators test..."
python test_indicators.py

echo "All tests completed!"