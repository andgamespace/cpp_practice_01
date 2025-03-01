//
// Improved DataLoader.hpp
// Created by anshc on 2/27/25.
// Updated for robustness and extensibility for a vectorized gym environment.
//
#ifndef DATALOADER_H
#define DATALOADER_H
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <arrow/api.h>

class DataLoader {
public:
    DataLoader() = default;
    ~DataLoader() = default;

    // Loads a ticker's CSV files (ordered from most recent to oldest) and builds an Arrow table.
    // Returns true if at least one file was processed successfully, false otherwise.
    bool loadTickerData(const std::string& ticker, const std::vector<std::string>& filePaths);

    // Retrieves the Arrow table for a given ticker symbol.
    std::shared_ptr<arrow::Table> getTickerData(const std::string& ticker) const;

private:
    // Internal mapping from ticker symbols to their corresponding Arrow tables.
    std::unordered_map<std::string, std::shared_ptr<arrow::Table>> tickerData_;

    // Helper function to impute missing values in a numeric column.
    // Missing values (NaN) are replaced with the average of up to 'window' preceding and 'window' succeeding valid entries.
    void imputeMissing(std::vector<double>& data, int window = 5);

    // Helper function to sort the merged data by datetime.
    void sortData(std::vector<std::string>& datetimes,
                  std::vector<double>& opens,
                  std::vector<double>& highs,
                  std::vector<double>& lows,
                  std::vector<double>& closes,
                  std::vector<double>& volumes);
};

#endif // DATALOADER_H
