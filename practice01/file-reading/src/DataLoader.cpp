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

bool DataLoader::loadTickerData(const std::string &ticker, const std::vector<std::string> &filePaths) {
    spdlog::info("Loading data for ticker: {}", ticker);
    std::vector<std::string> datetimes;
    std::vector<double> opens, highs, lows, closes, volumes;
    int filesProcessed = 0;

    for (const auto &filePath : filePaths) {
        if (!fs::exists(filePath)) {
            spdlog::error("File not found: {}", filePath);
            continue;
        }
        try {
            csv::CSVReader reader(filePath, csv::CSVFormat().delimiter(';').variable_columns(true));
            bool headerSkipped = false;
            std::vector<csv::CSVRow> rows;
            for (csv::CSVRow &row : reader) {
                if (!headerSkipped) {
                    // Skip header row
                    headerSkipped = true;
                    continue;
                }
                rows.push_back(row);
            }
            for (const auto &row : rows) {
                std::string datetime = row[0].get<>();
                double openVal   = std::stod(row[1].get<>());
                double highVal   = std::stod(row[2].get<>());
                double lowVal    = std::stod(row[3].get<>());
                double closeVal  = std::stod(row[4].get<>());
                double volumeVal = std::stod(row[5].get<>());
                datetimes.push_back(datetime);
                opens.push_back(openVal);
                highs.push_back(highVal);
                lows.push_back(lowVal);
                closes.push_back(closeVal);
                volumes.push_back(volumeVal);
            }
            ++filesProcessed;
        } catch (const std::exception &e) {
            spdlog::error("Failed to process file {}: {}", filePath, e.what());
            continue;
        }
    }

    if (filesProcessed == 0 || datetimes.empty()) {
        spdlog::error("No valid data loaded for ticker: {}", ticker);
        return false;
    }

    // Sort by datetime & impute missing
    sortData(datetimes, opens, highs, lows, closes, volumes);
    imputeMissing(opens);
    imputeMissing(highs);
    imputeMissing(lows);
    imputeMissing(closes);
    imputeMissing(volumes);

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
            spdlog::error("Error appending data at index {}", i);
            return false;
        }
    }

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

    auto schema = arrow::schema({
        arrow::field("datetime", arrow::utf8()),
        arrow::field("open", arrow::float64()),
        arrow::field("high", arrow::float64()),
        arrow::field("low", arrow::float64()),
        arrow::field("close", arrow::float64()),
        arrow::field("volume", arrow::float64())
    });

    auto table = arrow::Table::Make(schema,
        {dateArray, openArray, highArray, lowArray, closeArray, volumeArray});
    tickerData_[ticker] = table;
    spdlog::info("Successfully loaded {} rows for ticker {}", table->num_rows(), ticker);
    return true;
}

std::shared_ptr<arrow::Table> DataLoader::getTickerData(const std::string &ticker) const {
    auto it = tickerData_.find(ticker);
    return (it != tickerData_.end()) ? it->second : nullptr;
}
