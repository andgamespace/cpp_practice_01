#ifndef DATALOADER_H
#define DATALOADER_H
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <arrow/api.h>

/**
 * @brief DataLoader reads CSV files into Arrow tables for each ticker.
 */
class DataLoader {
public:
    DataLoader() : debug_(true) {}
    ~DataLoader() = default;

    // Load CSV files for a ticker and build an Arrow table.
    // Returns true if at least one file is processed successfully.
    bool loadTickerData(const std::string &ticker, const std::vector<std::string> &filePaths);

    // Retrieve the Arrow table for a given ticker.
    std::shared_ptr<arrow::Table> getTickerData(const std::string &ticker) const;

private:
    // Ticker -> Arrow table
    std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tickerData_;
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
};

#endif // DATALOADER_H
