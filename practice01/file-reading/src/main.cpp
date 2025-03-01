#include <iostream>
#include <vector>
#include <string>
#include "DataLoader.h"
#include <arrow/api.h>
#include <iomanip>

// Function to print the first few rows of the Arrow Table.
void printHead(const std::shared_ptr<arrow::Table>& table, int numRows = 5) {
    auto dateArray   = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
    auto openArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    auto highArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
    auto lowArray    = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
    auto closeArray  = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
    auto volumeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));

    int rows = static_cast<int>(table->num_rows());
    rows = std::min(rows, numRows);

    std::cout << std::left << std::setw(20) << "Datetime"
              << std::right << std::setw(10) << "Open"
              << std::setw(10) << "High"
              << std::setw(10) << "Low"
              << std::setw(10) << "Close"
              << std::setw(12) << "Volume" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (int i = 0; i < rows; i++) {
        std::cout << std::left << std::setw(20) << dateArray->GetString(i)
                  << std::right << std::setw(10) << openArray->Value(i)
                  << std::setw(10) << highArray->Value(i)
                  << std::setw(10) << lowArray->Value(i)
                  << std::setw(10) << closeArray->Value(i)
                  << std::setw(12) << volumeArray->Value(i) << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Update the base directory to point correctly relative to the build directory.
    // When running from "cmake-build-debug", "../src/stock_data/" points to the CSV folder.
    std::string baseDir = "../src/stock_data/";

    // List of tickers to process.
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};

    // Instantiate the DataLoader.
    DataLoader loader;

    // For each ticker, create a vector of file paths. Files are assumed ordered from most recent to oldest.
    for (const auto& ticker : tickers) {
        std::vector<std::string> filePaths;
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min.csv");
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min(1).csv");
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min(2).csv");

        std::cout << "=============================" << std::endl;
        std::cout << "Processing ticker: " << ticker << std::endl;
        std::cout << "=============================" << std::endl;

        if (!loader.loadTickerData(ticker, filePaths)) {
            std::cerr << "Failed to load data for ticker: " << ticker << std::endl;
            continue;
        }

        // Retrieve and print a preview of the Arrow table.
        auto table = loader.getTickerData(ticker);
        if (table) {
            std::cout << "Schema:\n" << table->schema()->ToString() << std::endl;
            std::cout << "Number of rows: " << table->num_rows() << std::endl;
            std::cout << "Showing first few rows:" << std::endl;
            printHead(table, 5);
        }
        std::cout << std::endl;
    }

    return 0;
}
