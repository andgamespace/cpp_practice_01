#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <csv.hpp> // vincentlaucsb/csv-parser
#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

namespace fs = std::filesystem;

void DataLoader::imputeMissing(std::vector<double> &data, int window) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i])) {
            double sum = 0.0;
            int count = 0;
            // Look backward up to 'window'
            for (int j = static_cast<int>(i) - window; j < static_cast<int>(i); ++j) {
                if (j >= 0 && !std::isnan(data[j])) {
                    sum += data[j];
                    ++count;
                }
            }
            // Look forward up to 'window'
            for (size_t j = i + 1; j < std::min(data.size(), i + window + 1); ++j) {
                if (!std::isnan(data[j])) {
                    sum += data[j];
                    ++count;
                }
            }
            data[i] = (count > 0) ? sum / count : 0.0;
        }
    }
}

void DataLoader::sortData(std::vector<std::string> &datetimes,
                           std::vector<double> &opens,
                           std::vector<double> &highs,
                           std::vector<double> &lows,
                           std::vector<double> &closes,
                           std::vector<double> &volumes) {
    std::vector<size_t> indices(datetimes.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    // Sort by datetime ascending (lex order).
    std::sort(indices.begin(), indices.end(), [&datetimes](size_t a, size_t b) {
        return datetimes[a] < datetimes[b];
    });
    // Reorder everything according to sorted indices.
    std::vector<std::string> sortedDates;
    std::vector<double> sortedOpens, sortedHighs, sortedLows, sortedCloses, sortedVolumes;
    sortedDates.reserve(indices.size());
    sortedOpens.reserve(indices.size());
    sortedHighs.reserve(indices.size());
    sortedLows.reserve(indices.size());
    sortedCloses.reserve(indices.size());
    sortedVolumes.reserve(indices.size());

    for (size_t idx : indices) {
        sortedDates.push_back(datetimes[idx]);
        sortedOpens.push_back(opens[idx]);
        sortedHighs.push_back(highs[idx]);
        sortedLows.push_back(lows[idx]);
        sortedCloses.push_back(closes[idx]);
        sortedVolumes.push_back(volumes[idx]);
    }
    datetimes = std::move(sortedDates);
    opens = std::move(sortedOpens);
    highs = std::move(sortedHighs);
    lows = std::move(sortedLows);
    closes = std::move(sortedCloses);
    volumes = std::move(sortedVolumes);
}

bool DataLoader::processCSVFile(const std::string &filePath,
                               std::vector<std::string> &datetimes,
                               std::vector<double> &opens,
                               std::vector<double> &highs,
                               std::vector<double> &lows,
                               std::vector<double> &closes,
                               std::vector<double> &volumes) {
    if (!fs::exists(filePath)) {
        if (debug_) spdlog::error("File not found: {}", filePath);
        return false;
    }
    
    try {
        // First try to detect the delimiter
        std::ifstream testFile(filePath);
        if (!testFile.is_open()) {
            if (debug_) spdlog::error("Could not open file: {}", filePath);
            return false;
        }
        
        std::string firstLine;
        std::getline(testFile, firstLine);
        testFile.close();
        
        char delimiter = ';';  // Default delimiter
        
        // Try to detect the delimiter
        if (firstLine.find(',') != std::string::npos) {
            delimiter = ',';
        } else if (firstLine.find(';') != std::string::npos) {
            delimiter = ';';
        } else if (firstLine.find('\t') != std::string::npos) {
            delimiter = '\t';
        }
        
        if (debug_) spdlog::info("Detected delimiter '{}' for file: {}", delimiter, filePath);
        
        // Now read the CSV with the detected delimiter
        csv::CSVReader reader(filePath, csv::CSVFormat().delimiter(delimiter).variable_columns(true));
        
        // Try to detect column indices
        std::vector<csv::CSVRow> rows;
        bool headerSkipped = false;
        int dateTimeCol = 0;
        int openCol = 1;
        int highCol = 2;
        int lowCol = 3;
        int closeCol = 4;
        int volumeCol = 5;
        
        // Read all rows
        for (csv::CSVRow &row : reader) {
            if (!headerSkipped) {
                // Try to detect column indices from header
                for (size_t i = 0; i < row.size(); ++i) {
                    std::string header = row[i].get<>();
                    std::transform(header.begin(), header.end(), header.begin(), ::tolower);
                    
                    if (header.find("date") != std::string::npos || header.find("time") != std::string::npos) {
                        dateTimeCol = i;
                    } else if (header.find("open") != std::string::npos) {
                        openCol = i;
                    } else if (header.find("high") != std::string::npos) {
                        highCol = i;
                    } else if (header.find("low") != std::string::npos) {
                        lowCol = i;
                    } else if (header.find("close") != std::string::npos) {
                        closeCol = i;
                    } else if (header.find("volume") != std::string::npos || header.find("vol") != std::string::npos) {
                        volumeCol = i;
                    }
                }
                
                if (debug_) {
                    spdlog::info("Detected columns for {}: datetime={}, open={}, high={}, low={}, close={}, volume={}",
                                filePath, dateTimeCol, openCol, highCol, lowCol, closeCol, volumeCol);
                }
                
                headerSkipped = true;
                continue;
            }
            rows.push_back(row);
        }
        
        // Process rows
        int validRows = 0;
        int skippedRows = 0;
        
        for (const auto &row : rows) {
            try {
                // Check if row has enough columns
                if (row.size() <= std::max({dateTimeCol, openCol, highCol, lowCol, closeCol, volumeCol})) {
                    if (debug_) spdlog::warn("Row has insufficient columns in {}", filePath);
                    skippedRows++;
                    continue;
                }
                
                std::string datetime = row[dateTimeCol].get<>();
                double openVal   = std::stod(row[openCol].get<>());
                double highVal   = std::stod(row[highCol].get<>());
                double lowVal    = std::stod(row[lowCol].get<>());
                double closeVal  = std::stod(row[closeCol].get<>());
                double volumeVal = std::stod(row[volumeCol].get<>());
                
                // Basic validation
                if (highVal < lowVal || openVal <= 0 || closeVal <= 0 || highVal <= 0 || lowVal <= 0) {
                    if (debug_) spdlog::warn("Invalid price values in {}", filePath);
                    skippedRows++;
                    continue;
                }
                
                datetimes.push_back(datetime);
                opens.push_back(openVal);
                highs.push_back(highVal);
                lows.push_back(lowVal);
                closes.push_back(closeVal);
                volumes.push_back(volumeVal);
                validRows++;
            } catch (const std::exception &e) {
                if (debug_) spdlog::warn("Skipping malformed row in {}: {}", filePath, e.what());
                skippedRows++;
                continue;
            }
        }
        
        if (debug_) spdlog::info("Processed {} rows from {}: {} valid, {} skipped",
                                rows.size(), filePath, validRows, skippedRows);
        
        return validRows > 0;
    } catch (const std::exception &e) {
        if (debug_) spdlog::error("Failed to process file {}: {}", filePath, e.what());
        return false;
    }
}

std::shared_ptr<arrow::Table> DataLoader::createArrowTable(
    const std::vector<std::string> &datetimes,
    const std::vector<double> &opens,
    const std::vector<double> &highs,
    const std::vector<double> &lows,
    const std::vector<double> &closes,
    const std::vector<double> &volumes) {
    
    // Build Arrow arrays
    arrow::StringBuilder dateBuilder;
    arrow::DoubleBuilder openBuilder, highBuilder, lowBuilder, closeBuilder, volumeBuilder;

    for (size_t i = 0; i < datetimes.size(); ++i) {
        if (!dateBuilder.Append(datetimes[i]).ok() ||
            !openBuilder.Append(opens[i]).ok() ||
            !highBuilder.Append(highs[i]).ok() ||
            !lowBuilder.Append(lows[i]).ok() ||
            !closeBuilder.Append(closes[i]).ok() ||
            !volumeBuilder.Append(volumes[i]).ok()) {
            if (debug_) spdlog::error("Error appending data at index {}", i);
            return nullptr;
        }
    }

    std::shared_ptr<arrow::Array> dateArray, openArray, highArray, lowArray, closeArray, volumeArray;
    if (!dateBuilder.Finish(&dateArray).ok() ||
        !openBuilder.Finish(&openArray).ok() ||
        !highBuilder.Finish(&highArray).ok() ||
        !lowBuilder.Finish(&lowArray).ok() ||
        !closeBuilder.Finish(&closeArray).ok() ||
        !volumeBuilder.Finish(&volumeArray).ok()) {
        if (debug_) spdlog::error("Error finalizing Arrow arrays");
        return nullptr;
    }

    auto schema = arrow::schema({
        arrow::field("datetime", arrow::utf8()),
        arrow::field("open", arrow::float64()),
        arrow::field("high", arrow::float64()),
        arrow::field("low", arrow::float64()),
        arrow::field("close", arrow::float64()),
        arrow::field("volume", arrow::float64())
    });

    return arrow::Table::Make(schema,
        {dateArray, openArray, highArray, lowArray, closeArray, volumeArray});
}

bool DataLoader::loadTickerData(const std::string &ticker, const std::vector<std::string> &filePaths) {
    if (debug_) spdlog::info("Loading data for ticker: {}", ticker);
    std::vector<std::string> datetimes;
    std::vector<double> opens, highs, lows, closes, volumes;
    int filesProcessed = 0;

    for (const auto &filePath : filePaths) {
        if (processCSVFile(filePath, datetimes, opens, highs, lows, closes, volumes)) {
            ++filesProcessed;
        }
    }

    if (filesProcessed == 0 || datetimes.empty()) {
        if (debug_) spdlog::error("No valid data loaded for ticker: {}", ticker);
        return false;
    }

    // Sort by datetime & impute missing
    sortData(datetimes, opens, highs, lows, closes, volumes);
    imputeMissing(opens);
    imputeMissing(highs);
    imputeMissing(lows);
    imputeMissing(closes);
    imputeMissing(volumes);

    auto table = createArrowTable(datetimes, opens, highs, lows, closes, volumes);
    if (!table) {
        if (debug_) spdlog::error("Failed to create Arrow table for ticker: {}", ticker);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(tickerDataMutex_);
        tickerData_[ticker] = table;
    }
    
    if (debug_) spdlog::info("Successfully loaded {} rows for ticker {}", table->num_rows(), ticker);
    return true;
}

int DataLoader::loadMultipleTickers(const std::unordered_map<std::string, std::vector<std::string>> &tickerFilePaths) {
    if (debug_) spdlog::info("Loading data for {} tickers concurrently", tickerFilePaths.size());
    
    tf::Executor executor;
    tf::Taskflow taskflow;
    
    std::atomic<int> successCount = 0;
    
    // Create a task for each ticker
    for (const auto &[ticker, filePaths] : tickerFilePaths) {
        taskflow.emplace([this, ticker, filePaths, &successCount]() {
            if (this->loadTickerData(ticker, filePaths)) {
                successCount++;
            }
        });
    }
    
    executor.run(taskflow).wait();
    
    if (debug_) spdlog::info("Successfully loaded {} out of {} tickers", successCount.load(), tickerFilePaths.size());
    return successCount;
}

std::shared_ptr<arrow::Table> DataLoader::getTickerData(const std::string &ticker) const {
    std::lock_guard<std::mutex> lock(tickerDataMutex_);
    auto it = tickerData_.find(ticker);
    return (it != tickerData_.end()) ? it->second : nullptr;
}

std::unordered_map<std::string, std::shared_ptr<arrow::Table>> DataLoader::getAllTickerData() const {
    std::lock_guard<std::mutex> lock(tickerDataMutex_);
    return tickerData_;
}

bool DataLoader::hasTickerData(const std::string &ticker) const {
    std::lock_guard<std::mutex> lock(tickerDataMutex_);
    return tickerData_.find(ticker) != tickerData_.end();
}

std::vector<std::string> DataLoader::getAvailableTickers() const {
    std::lock_guard<std::mutex> lock(tickerDataMutex_);
    std::vector<std::string> tickers;
    tickers.reserve(tickerData_.size());
    
    for (const auto &pair : tickerData_) {
        tickers.push_back(pair.first);
    }
    
    return tickers;
}

void DataLoader::updateTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table) {
    if (debug_) spdlog::info("Updating ticker data for {}", ticker);
    
    std::lock_guard<std::mutex> lock(tickerDataMutex_);
    tickerData_[ticker] = table;
}
