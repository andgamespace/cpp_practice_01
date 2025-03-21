#ifndef DATALOADER_H
#define DATALOADER_H
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <future>
#include <arrow/api.h>
#include <taskflow/taskflow.hpp>

/**
 * @brief DataLoader reads CSV files into Arrow tables for each ticker.
 * Uses Taskflow for concurrent data loading and processing.
 */
class DataLoader {
public:
    DataLoader() : debug_(false) {}
    ~DataLoader() = default;

    // Load CSV files for a ticker and build an Arrow table.
    // Returns true if at least one file is processed successfully.
    bool loadTickerData(const std::string &ticker, const std::vector<std::string> &filePaths);

    // Load multiple tickers concurrently using Taskflow
    // Returns the number of successfully loaded tickers
    int loadMultipleTickers(const std::unordered_map<std::string, std::vector<std::string>> &tickerFilePaths);

    // Retrieve the Arrow table for a given ticker.
    std::shared_ptr<arrow::Table> getTickerData(const std::string &ticker) const;

    // Get all loaded ticker data
    std::unordered_map<std::string, std::shared_ptr<arrow::Table>> getAllTickerData() const;

    // Check if a ticker is loaded
    bool hasTickerData(const std::string &ticker) const;

    // Get available tickers
    std::vector<std::string> getAvailableTickers() const;
    
    // Update ticker data with a new table
    void updateTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table);

    // Enable/disable debug logging
    void setDebugMode(bool enable) { debug_ = enable; }

private:
    // Ticker -> Arrow table
    std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tickerData_;
    mutable std::mutex tickerDataMutex_;
    bool debug_;

    // Helper to impute missing numeric data.
    void imputeMissing(std::vector<double> &data, int window = 5);

    // Helper to sort data by datetime.
    void sortData(std::vector<std::string> &datetimes,
                   std::vector<double> &opens,
                   std::vector<double> &highs,
                   std::vector<double> &lows,
                   std::vector<double> &closes,
                   std::vector<double> &volumes);

    // Process a single CSV file and return the data
    bool processCSVFile(const std::string &filePath,
                        std::vector<std::string> &datetimes,
                        std::vector<double> &opens,
                        std::vector<double> &highs,
                        std::vector<double> &lows,
                        std::vector<double> &closes,
                        std::vector<double> &volumes);

    // Create Arrow table from processed data
    std::shared_ptr<arrow::Table> createArrowTable(
        const std::vector<std::string> &datetimes,
        const std::vector<double> &opens,
        const std::vector<double> &highs,
        const std::vector<double> &lows,
        const std::vector<double> &closes,
        const std::vector<double> &volumes);
};

#endif // DATALOADER_H
