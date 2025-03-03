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

namespace fs = std::filesystem;

// Helper: impute missing (NaN) values in a numeric column vector.
void DataLoader::imputeMissing(std::vector<double>& data, int window) {
    for (size_t i = 0; i < data.size(); ++i) {
        if (std::isnan(data[i])) {
            double sum = 0.0;
            int count = 0;
            // Look backwards up to 'window' rows.
            for (int j = static_cast<int>(i) - window; j < static_cast<int>(i); ++j) {
                if (j >= 0 && !std::isnan(data[j])) {
                    sum += data[j];
                    ++count;
                }
            }
            // Look forwards up to 'window' rows.
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

// Helper: sort merged data by datetime.
void DataLoader::sortData(std::vector<std::string>& datetimes,
                          std::vector<double>& opens,
                          std::vector<double>& highs,
                          std::vector<double>& lows,
                          std::vector<double>& closes,
                          std::vector<double>& volumes) {
    std::vector<size_t> indices(datetimes.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    // The datetime format "YYYY-MM-DD HH:MM:SS" allows lexicographical sorting.
    std::sort(indices.begin(), indices.end(), [&datetimes](size_t a, size_t b) {
        return datetimes[a] < datetimes[b];
    });

    // Create copies of the original data.
    std::vector<std::string> sortedDates;
    std::vector<double> sortedOpens, sortedHighs, sortedLows, sortedCloses, sortedVolumes;
    for (size_t idx : indices) {
        sortedDates.push_back(datetimes[idx]);
        sortedOpens.push_back(opens[idx]);
        sortedHighs.push_back(highs[idx]);
        sortedLows.push_back(lows[idx]);
        sortedCloses.push_back(closes[idx]);
        sortedVolumes.push_back(volumes[idx]);
    }

    // Replace original vectors with sorted data.
    datetimes = std::move(sortedDates);
    opens = std::move(sortedOpens);
    highs = std::move(sortedHighs);
    lows = std::move(sortedLows);
    closes = std::move(sortedCloses);
    volumes = std::move(sortedVolumes);
}

// Loads CSV files for a given ticker symbol and creates an Arrow Table.
bool DataLoader::loadTickerData(const std::string& ticker, const std::vector<std::string>& filePaths) {
    spdlog::info("Loading data for ticker: {}", ticker);

    // Vectors to store merged data from all files.
    std::vector<std::string> datetimes;
    std::vector<double> opens, highs, lows, closes, volumes;

    int filesProcessed = 0;

    for (const auto& filePath : filePaths) {
        if (!fs::exists(filePath)) {
            spdlog::error("File not found: {}", filePath);
            continue;
        }

        try {
            csv::CSVReader reader(filePath, csv::CSVFormat().delimiter(';').variable_columns(true));
            bool headerSkipped = false;
            std::vector<csv::CSVRow> rows;
            for (csv::CSVRow& row : reader) {
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }
                rows.push_back(row);
            }

            // Process rows sequentially using a simple for loop.
            for (const auto& row : rows) {
                std::string datetime = row[0].template get<std::string>();
                double openVal = std::numeric_limits<double>::quiet_NaN();
                double highVal = std::numeric_limits<double>::quiet_NaN();
                double lowVal  = std::numeric_limits<double>::quiet_NaN();
                double closeVal= std::numeric_limits<double>::quiet_NaN();
                double volumeVal = std::numeric_limits<double>::quiet_NaN();

                try {
                    openVal = std::stod(row[1].template get<std::string>());
                } catch (const std::exception& e) {
                    spdlog::error("Error parsing open value in file {}: {}", filePath, e.what());
                    continue; // skip this row
                }
                try {
                    highVal = std::stod(row[2].template get<std::string>());
                } catch (const std::exception& e) {
                    spdlog::error("Error parsing high value in file {}: {}", filePath, e.what());
                    continue;
                }
                try {
                    lowVal = std::stod(row[3].template get<std::string>());
                } catch (const std::exception& e) {
                    spdlog::error("Error parsing low value in file {}: {}", filePath, e.what());
                    continue;
                }
                try {
                    closeVal = std::stod(row[4].template get<std::string>());
                } catch (const std::exception& e) {
                    spdlog::error("Error parsing close value in file {}: {}", filePath, e.what());
                    continue;
                }
                try {
                    volumeVal = std::stod(row[5].template get<std::string>());
                } catch (const std::exception& e) {
                    spdlog::error("Error parsing volume value in file {}: {}", filePath, e.what());
                    continue;
                }

                // Append parsed values to vectors.
                datetimes.emplace_back(datetime);
                opens.emplace_back(openVal);
                highs.emplace_back(highVal);
                lows.emplace_back(lowVal);
                closes.emplace_back(closeVal);
                volumes.emplace_back(volumeVal);
            }
            ++filesProcessed;
        } catch (const std::exception& e) {
            spdlog::error("Failed to process file {}: {}", filePath, e.what());
            continue;
        }
    }

    if (filesProcessed == 0 || datetimes.empty()) {
        spdlog::error("No valid data loaded for ticker: {}", ticker);
        return false;
    }

    // Sort data by datetime to ensure chronological order.
    sortData(datetimes, opens, highs, lows, closes, volumes);

    // Impute missing numeric data for each column.
    imputeMissing(opens);
    imputeMissing(highs);
    imputeMissing(lows);
    imputeMissing(closes);
    imputeMissing(volumes);

    // Build Arrow arrays using builders.
    arrow::StringBuilder dateBuilder;
    arrow::DoubleBuilder openBuilder;
    arrow::DoubleBuilder highBuilder;
    arrow::DoubleBuilder lowBuilder;
    arrow::DoubleBuilder closeBuilder;
    arrow::DoubleBuilder volumeBuilder;

    for (size_t i = 0; i < datetimes.size(); ++i) {
        if (!dateBuilder.Append(datetimes[i]).ok()) {
            spdlog::error("Error appending datetime at index {}", i);
            return false;
        }
        if (!openBuilder.Append(opens[i]).ok() ||
            !highBuilder.Append(highs[i]).ok() ||
            !lowBuilder.Append(lows[i]).ok() ||
            !closeBuilder.Append(closes[i]).ok() ||
            !volumeBuilder.Append(volumes[i]).ok()) {
            spdlog::error("Error appending numeric data at index {}", i);
            return false;
        }
    }

    // Finalize the builders to get Arrow arrays.
    std::shared_ptr<arrow::Array> dateArray, openArray, highArray, lowArray, closeArray, volumeArray;
    if (!dateBuilder.Finish(&dateArray).ok() ||
        !openBuilder.Finish(&openArray).ok() ||
        !highBuilder.Finish(&highArray).ok() ||
        !lowBuilder.Finish(&lowArray).ok() ||
        !closeBuilder.Finish(&closeArray).ok() ||
        !volumeBuilder.Finish(&volumeArray).ok()) {
        spdlog::error("Error finalizing Arrow arrays for ticker: {}", ticker);
        return false;
    }

    // Create an Arrow schema and table.
    auto schema = arrow::schema({
        arrow::field("datetime", arrow::utf8()),
        arrow::field("open", arrow::float64()),
        arrow::field("high", arrow::float64()),
        arrow::field("low", arrow::float64()),
        arrow::field("close", arrow::float64()),
        arrow::field("volume", arrow::float64())
    });

    std::shared_ptr<arrow::Table> table = arrow::Table::Make(
        schema, {dateArray, openArray, highArray, lowArray, closeArray, volumeArray});

    tickerData_[ticker] = table;
    spdlog::info("Successfully loaded {} rows for ticker {}", table->num_rows(), ticker);
    return true;
}

std::shared_ptr<arrow::Table> DataLoader::getTickerData(const std::string& ticker) const {
    auto it = tickerData_.find(ticker);
    return (it != tickerData_.end()) ? it->second : nullptr;
}
