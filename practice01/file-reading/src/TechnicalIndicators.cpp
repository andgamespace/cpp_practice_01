#include "TechnicalIndicators.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

TechnicalIndicators::TechnicalIndicators() {
    // Initialize Arrow compute context
}

void TechnicalIndicators::setDebugMode(bool debug) {
    debug_ = debug;
}

std::shared_ptr<arrow::Table> TechnicalIndicators::calculateSMA(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating SMA for column {} with period {}", column, period);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate the moving average
    auto maArray = calculateMovingAverage(array, period);
    
    // Handle edge cases (fill NaN values at the beginning)
    auto resultArray = handleEdgeCases(maArray, period);
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or(column + "_sma_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, resultArray);
}

std::shared_ptr<arrow::Table> TechnicalIndicators::calculateEMA(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating EMA for column {} with period {}", column, period);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate the exponential moving average
    auto emaArray = calculateExponentialMovingAverage(array, period);
    
    // Handle edge cases (fill NaN values at the beginning)
    auto resultArray = handleEdgeCases(emaArray, period);
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or(column + "_ema_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, resultArray);
}

std::shared_ptr<arrow::Table> TechnicalIndicators::calculateRSI(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating RSI for column {} with period {}", column, period);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate price changes
    arrow::DoubleBuilder builder;
    builder.Reserve(array->length());
    
    // First value has no change
    builder.Append(0.0);
    
    // Calculate changes for the rest
    for (int64_t i = 1; i < array->length(); ++i) {
        double change = array->Value(i) - array->Value(i - 1);
        builder.Append(change);
    }
    
    std::shared_ptr<arrow::DoubleArray> changesArray;
    ARROW_RETURN_NOT_OK(builder.Finish(&changesArray));
    
    // Calculate gains and losses
    arrow::DoubleBuilder gainsBuilder, lossesBuilder;
    gainsBuilder.Reserve(changesArray->length());
    lossesBuilder.Reserve(changesArray->length());
    
    for (int64_t i = 0; i < changesArray->length(); ++i) {
        double change = changesArray->Value(i);
        gainsBuilder.Append(std::max(0.0, change));
        lossesBuilder.Append(std::max(0.0, -change));
    }
    
    std::shared_ptr<arrow::DoubleArray> gainsArray, lossesArray;
    ARROW_RETURN_NOT_OK(gainsBuilder.Finish(&gainsArray));
    ARROW_RETURN_NOT_OK(lossesBuilder.Finish(&lossesArray));
    
    // Calculate average gains and losses
    auto avgGainsArray = calculateExponentialMovingAverage(gainsArray, period);
    auto avgLossesArray = calculateExponentialMovingAverage(lossesArray, period);
    
    // Calculate RS and RSI
    arrow::DoubleBuilder rsiBuilder;
    rsiBuilder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        if (i < period) {
            // Not enough data for the period
            rsiBuilder.Append(50.0); // Default value for the beginning
        } else {
            double avgGain = avgGainsArray->Value(i);
            double avgLoss = avgLossesArray->Value(i);
            
            if (avgLoss == 0.0) {
                rsiBuilder.Append(100.0);
            } else {
                double rs = avgGain / avgLoss;
                double rsi = 100.0 - (100.0 / (1.0 + rs));
                rsiBuilder.Append(rsi);
            }
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> rsiArray;
    ARROW_RETURN_NOT_OK(rsiBuilder.Finish(&rsiArray));
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or(column + "_rsi_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, rsiArray);
}

std::shared_ptr<arrow::Table> TechnicalIndicators::calculateBollingerBands(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    double stdDev,
    std::optional<std::string> upperBandName,
    std::optional<std::string> middleBandName,
    std::optional<std::string> lowerBandName) {
    
    if (debug_) spdlog::info("Calculating Bollinger Bands for column {} with period {} and stdDev {}", 
                           column, period, stdDev);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate the SMA (middle band)
    auto smaArray = calculateMovingAverage(array, period);
    
    // Calculate standard deviation
    auto stdDevArray = calculateStdDev(table, column, period)->column(table->num_columns() - 1)->chunk(0);
    
    // Calculate upper and lower bands
    arrow::DoubleBuilder upperBuilder, lowerBuilder;
    upperBuilder.Reserve(array->length());
    lowerBuilder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        if (i < period - 1) {
            // Not enough data for the period
            upperBuilder.Append(array->Value(i));
            lowerBuilder.Append(array->Value(i));
        } else {
            double sma = smaArray->Value(i);
            double sd = std::static_pointer_cast<arrow::DoubleArray>(stdDevArray)->Value(i);
            
            upperBuilder.Append(sma + stdDev * sd);
            lowerBuilder.Append(sma - stdDev * sd);
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> upperArray, lowerArray;
    ARROW_RETURN_NOT_OK(upperBuilder.Finish(&upperArray));
    ARROW_RETURN_NOT_OK(lowerBuilder.Finish(&lowerArray));
    
    // Generate column names if not provided
    std::string upperName = upperBandName.value_or(column + "_bband_upper");
    std::string middleName = middleBandName.value_or(column + "_bband_middle");
    std::string lowerName = lowerBandName.value_or(column + "_bband_lower");
    
    // Add the new columns to the table
    auto resultTable = addColumn(table, middleName, smaArray);
    resultTable = addColumn(resultTable, upperName, upperArray);
    resultTable = addColumn(resultTable, lowerName, lowerArray);
    
    return resultTable;
}

std::shared_ptr<arrow::Table> TechnicalIndicators::calculateMACD(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int fastPeriod,
    int slowPeriod,
    int signalPeriod,
    std::optional<std::string> macdName,
    std::optional<std::string> signalName,
    std::optional<std::string> histogramName) {
    
    if (debug_) spdlog::info("Calculating MACD for column {} with fastPeriod {}, slowPeriod {}, signalPeriod {}", 
                           column, fastPeriod, slowPeriod, signalPeriod);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate fast and slow EMAs
    auto fastEMA = calculateExponentialMovingAverage(array, fastPeriod);
    auto slowEMA = calculateExponentialMovingAverage(array, slowPeriod);
    
    // Calculate MACD line
    arrow::DoubleBuilder macdBuilder;
    macdBuilder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        double macdValue = fastEMA->Value(i) - slowEMA->Value(i);
        macdBuilder.Append(macdValue);
    }
    
    std::shared_ptr<arrow::DoubleArray> macdArray;
    ARROW_RETURN_NOT_OK(macdBuilder.Finish(&macdArray));
    
    // Calculate signal line (EMA of MACD)
    auto signalArray = calculateExponentialMovingAverage(macdArray, signalPeriod);
    
    // Calculate histogram (MACD - Signal)
    arrow::DoubleBuilder histBuilder;
    histBuilder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        double histValue = macdArray->Value(i) - signalArray->Value(i);
        histBuilder.Append(histValue);
    }
    
    std::shared_ptr<arrow::DoubleArray> histArray;
    ARROW_RETURN_NOT_OK(histBuilder.Finish(&histArray));
    
    // Generate column names if not provided
    std::string macdColName = macdName.value_or(column + "_macd");
    std::string signalColName = signalName.value_or(column + "_macd_signal");
    std::string histColName = histogramName.value_or(column + "_macd_histogram");
    
    // Add the new columns to the table
    auto resultTable = addColumn(table, macdColName, macdArray);
    resultTable = addColumn(resultTable, signalColName, signalArray);
    resultTable = addColumn(resultTable, histColName, histArray);
    
    return resultTable;
}

// Helper function to calculate moving average
std::shared_ptr<arrow::Array> TechnicalIndicators::calculateMovingAverage(
    const std::shared_ptr<arrow::Array>& array,
    int period) {
    
    auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
    arrow::DoubleBuilder builder;
    builder.Reserve(doubleArray->length());
    
    for (int64_t i = 0; i < doubleArray->length(); ++i) {
        if (i < period - 1) {
            // Not enough data for the period
            builder.Append(doubleArray->Value(i));
        } else {
            // Calculate sum for the period
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += doubleArray->Value(i - j);
            }
            
            // Calculate average
            double average = sum / period;
            builder.Append(average);
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    
    return result;
}

// Helper function to calculate exponential moving average
std::shared_ptr<arrow::Array> TechnicalIndicators::calculateExponentialMovingAverage(
    const std::shared_ptr<arrow::Array>& array,
    int period) {
    
    auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
    arrow::DoubleBuilder builder;
    builder.Reserve(doubleArray->length());
    
    // Calculate multiplier
    double multiplier = 2.0 / (period + 1.0);
    
    // First value is SMA
    double sma = 0.0;
    for (int i = 0; i < period && i < doubleArray->length(); ++i) {
        sma += doubleArray->Value(i);
    }
    sma /= period;
    
    // First EMA is SMA
    double ema = sma;
    builder.Append(ema);
    
    // Calculate EMA for the rest
    for (int64_t i = 1; i < doubleArray->length(); ++i) {
        ema = (doubleArray->Value(i) - ema) * multiplier + ema;
        builder.Append(ema);
    }
    
    std::shared_ptr<arrow::DoubleArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    
    return result;
}

// Helper function to handle edge cases
std::shared_ptr<arrow::Array> TechnicalIndicators::handleEdgeCases(
    const std::shared_ptr<arrow::Array>& array,
    int period,
    double defaultValue) {
    
    auto doubleArray = std::static_pointer_cast<arrow::DoubleArray>(array);
    arrow::DoubleBuilder builder;
    builder.Reserve(doubleArray->length());
    
    for (int64_t i = 0; i < doubleArray->length(); ++i) {
        if (i < period - 1) {
            // Fill with default value for the beginning
            builder.Append(defaultValue);
        } else {
            builder.Append(doubleArray->Value(i));
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    
    return result;
}

// Helper function to add a column to a table
std::shared_ptr<arrow::Table> TechnicalIndicators::addColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& name,
    const std::shared_ptr<arrow::Array>& array) {
    
    // Create field for the new column
    auto field = arrow::field(name, arrow::float64());
    
    // Create schema for the new table
    auto oldSchema = table->schema();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    
    // Copy existing fields
    for (int i = 0; i < oldSchema->num_fields(); ++i) {
        fields.push_back(oldSchema->field(i));
    }
    
    // Add new field
    fields.push_back(field);
    auto newSchema = arrow::schema(fields);
    
    // Create arrays for the new table
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    
    // Copy existing arrays
    for (int i = 0; i < table->num_columns(); ++i) {
        arrays.push_back(table->column(i)->chunk(0));
    }
    
    // Add new array
    arrays.push_back(array);
    
    // Create new table
    return arrow::Table::Make(newSchema, arrays);
}

// Implementation of True Range calculation
std::shared_ptr<arrow::Array> TechnicalIndicators::calculateTrueRange(
    const std::shared_ptr<arrow::Array>& highArray,
    const std::shared_ptr<arrow::Array>& lowArray,
    const std::shared_ptr<arrow::Array>& closeArray) {
    
    auto highDoubleArray = std::static_pointer_cast<arrow::DoubleArray>(highArray);
    auto lowDoubleArray = std::static_pointer_cast<arrow::DoubleArray>(lowArray);
    auto closeDoubleArray = std::static_pointer_cast<arrow::DoubleArray>(closeArray);
    
    arrow::DoubleBuilder builder;
    builder.Reserve(highDoubleArray->length());
    
    // First value is just high - low
    builder.Append(highDoubleArray->Value(0) - lowDoubleArray->Value(0));
    
    // Calculate TR for the rest
    for (int64_t i = 1; i < highDoubleArray->length(); ++i) {
        double prevClose = closeDoubleArray->Value(i - 1);
        double high = highDoubleArray->Value(i);
        double low = lowDoubleArray->Value(i);
        
        double range1 = high - low;
        double range2 = std::abs(high - prevClose);
        double range3 = std::abs(low - prevClose);
        
        double tr = std::max({range1, range2, range3});
        builder.Append(tr);
    }
    
    std::shared_ptr<arrow::DoubleArray> result;
    ARROW_RETURN_NOT_OK(builder.Finish(&result));
    
    return result;
}

// Implementation of ATR calculation
std::shared_ptr<arrow::Table> TechnicalIndicators::calculateATR(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& highColumn,
    const std::string& lowColumn,
    const std::string& closeColumn,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating ATR with period {}", period);
    
    // Get the columns
    int highIndex = table->schema()->GetFieldIndex(highColumn);
    int lowIndex = table->schema()->GetFieldIndex(lowColumn);
    int closeIndex = table->schema()->GetFieldIndex(closeColumn);
    
    if (highIndex == -1 || lowIndex == -1 || closeIndex == -1) {
        if (debug_) spdlog::error("One or more required columns not found");
        throw std::runtime_error("Required columns not found");
    }
    
    auto highArray = table->column(highIndex)->chunk(0);
    auto lowArray = table->column(lowIndex)->chunk(0);
    auto closeArray = table->column(closeIndex)->chunk(0);
    
    // Calculate True Range
    auto trArray = calculateTrueRange(highArray, lowArray, closeArray);
    
    // Calculate ATR (EMA of TR)
    auto atrArray = calculateExponentialMovingAverage(trArray, period);
    
    // Handle edge cases
    auto resultArray = handleEdgeCases(atrArray, period);
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or("atr_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, resultArray);
}

// Implementation of Standard Deviation calculation
std::shared_ptr<arrow::Table> TechnicalIndicators::calculateStdDev(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating Standard Deviation for column {} with period {}", column, period);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate moving average
    auto maArray = calculateMovingAverage(array, period);
    
    // Calculate standard deviation
    arrow::DoubleBuilder builder;
    builder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        if (i < period - 1) {
            // Not enough data for the period
            builder.Append(0.0);
        } else {
            // Calculate sum of squared differences
            double ma = maArray->Value(i);
            double sumSquaredDiff = 0.0;
            
            for (int j = 0; j < period; ++j) {
                double diff = array->Value(i - j) - ma;
                sumSquaredDiff += diff * diff;
            }
            
            // Calculate standard deviation
            double stdDev = std::sqrt(sumSquaredDiff / period);
            builder.Append(stdDev);
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> stdDevArray;
    ARROW_RETURN_NOT_OK(builder.Finish(&stdDevArray));
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or(column + "_stddev_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, stdDevArray);
}

// Implementation of ROC calculation
std::shared_ptr<arrow::Table> TechnicalIndicators::calculateROC(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& column,
    int period,
    std::optional<std::string> newColumnName) {
    
    if (debug_) spdlog::info("Calculating ROC for column {} with period {}", column, period);
    
    // Get the column
    int columnIndex = table->schema()->GetFieldIndex(column);
    if (columnIndex == -1) {
        if (debug_) spdlog::error("Column {} not found in table", column);
        throw std::runtime_error("Column not found: " + column);
    }
    
    auto array = std::static_pointer_cast<arrow::DoubleArray>(table->column(columnIndex)->chunk(0));
    
    // Calculate ROC
    arrow::DoubleBuilder builder;
    builder.Reserve(array->length());
    
    for (int64_t i = 0; i < array->length(); ++i) {
        if (i < period) {
            // Not enough data for the period
            builder.Append(0.0);
        } else {
            double currentPrice = array->Value(i);
            double previousPrice = array->Value(i - period);
            
            if (previousPrice == 0.0) {
                builder.Append(0.0);
            } else {
                double roc = ((currentPrice - previousPrice) / previousPrice) * 100.0;
                builder.Append(roc);
            }
        }
    }
    
    std::shared_ptr<arrow::DoubleArray> rocArray;
    ARROW_RETURN_NOT_OK(builder.Finish(&rocArray));
    
    // Generate column name if not provided
    std::string colName = newColumnName.value_or(column + "_roc_" + std::to_string(period));
    
    // Add the new column to the table
    return addColumn(table, colName, rocArray);
}

// Implementation of multiple indicators calculation
std::shared_ptr<arrow::Table> TechnicalIndicators::calculateMultipleIndicators(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<std::string>& indicators,
    const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& params) {
    
    if (debug_) spdlog::info("Calculating multiple indicators: {}", indicators.size());
    
    auto resultTable = table;
    
    for (const auto& indicator : indicators) {
        if (indicator == "sma") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateSMA(resultTable, column, period);
            }
        } else if (indicator == "ema") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateEMA(resultTable, column, period);
            }
        } else if (indicator == "rsi") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateRSI(resultTable, column, period);
            }
        } else if (indicator == "bollinger") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                double stdDev = std::stod(p->second.at("stdDev"));
                resultTable = calculateBollingerBands(resultTable, column, period, stdDev);
            }
        } else if (indicator == "macd") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int fastPeriod = std::stoi(p->second.at("fastPeriod"));
                int slowPeriod = std::stoi(p->second.at("slowPeriod"));
                int signalPeriod = std::stoi(p->second.at("signalPeriod"));
                resultTable = calculateMACD(resultTable, column, fastPeriod, slowPeriod, signalPeriod);
            }
        } else if (indicator == "atr") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string highColumn = p->second.at("highColumn");
                std::string lowColumn = p->second.at("lowColumn");
                std::string closeColumn = p->second.at("closeColumn");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateATR(resultTable, highColumn, lowColumn, closeColumn, period);
            }
        } else if (indicator == "roc") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateROC(resultTable, column, period);
            }
        } else if (indicator == "stddev") {
            auto p = params.find(indicator);
            if (p != params.end()) {
                std::string column = p->second.at("column");
                int period = std::stoi(p->second.at("period"));
                resultTable = calculateStdDev(resultTable, column, period);
            }
        }
        // Add more indicators as needed
    }
    
    return resultTable;
}