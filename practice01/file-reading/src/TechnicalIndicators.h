#ifndef TECHNICAL_INDICATORS_H
#define TECHNICAL_INDICATORS_H
#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/api_vector.h>
#include <arrow/compute/api_aggregate.h>
#include <arrow/util/logging.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <optional>

/**
 * @brief Class for calculating technical indicators on Arrow tables
 * 
 * This class provides methods for calculating various technical indicators
 * on Arrow tables, which can be used for feature engineering in trading
 * strategies.
 */
class TechnicalIndicators {
public:
    /**
     * @brief Constructor
     */
    TechnicalIndicators();
    
    /**
     * @brief Set debug mode
     * 
     * @param debug Whether to enable debug logging
     */
    void setDebugMode(bool debug);
    
    /**
     * @brief Calculate Simple Moving Average (SMA)
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate SMA on
     * @param period Period for the moving average
     * @param newColumnName Name for the new column (default: "{column}_sma_{period}")
     * @return std::shared_ptr<arrow::Table> Table with SMA column added
     */
    std::shared_ptr<arrow::Table> calculateSMA(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Exponential Moving Average (EMA)
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate EMA on
     * @param period Period for the moving average
     * @param newColumnName Name for the new column (default: "{column}_ema_{period}")
     * @return std::shared_ptr<arrow::Table> Table with EMA column added
     */
    std::shared_ptr<arrow::Table> calculateEMA(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Relative Strength Index (RSI)
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate RSI on
     * @param period Period for RSI calculation
     * @param newColumnName Name for the new column (default: "{column}_rsi_{period}")
     * @return std::shared_ptr<arrow::Table> Table with RSI column added
     */
    std::shared_ptr<arrow::Table> calculateRSI(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Bollinger Bands
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate Bollinger Bands on
     * @param period Period for the moving average
     * @param stdDev Number of standard deviations for the bands
     * @param upperBandName Name for the upper band column (default: "{column}_bband_upper")
     * @param middleBandName Name for the middle band column (default: "{column}_bband_middle")
     * @param lowerBandName Name for the lower band column (default: "{column}_bband_lower")
     * @return std::shared_ptr<arrow::Table> Table with Bollinger Bands columns added
     */
    std::shared_ptr<arrow::Table> calculateBollingerBands(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        double stdDev = 2.0,
        std::optional<std::string> upperBandName = std::nullopt,
        std::optional<std::string> middleBandName = std::nullopt,
        std::optional<std::string> lowerBandName = std::nullopt);
    
    /**
     * @brief Calculate Moving Average Convergence Divergence (MACD)
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate MACD on
     * @param fastPeriod Period for the fast EMA
     * @param slowPeriod Period for the slow EMA
     * @param signalPeriod Period for the signal line
     * @param macdName Name for the MACD column (default: "{column}_macd")
     * @param signalName Name for the signal column (default: "{column}_macd_signal")
     * @param histogramName Name for the histogram column (default: "{column}_macd_histogram")
     * @return std::shared_ptr<arrow::Table> Table with MACD columns added
     */
    std::shared_ptr<arrow::Table> calculateMACD(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int fastPeriod = 12,
        int slowPeriod = 26,
        int signalPeriod = 9,
        std::optional<std::string> macdName = std::nullopt,
        std::optional<std::string> signalName = std::nullopt,
        std::optional<std::string> histogramName = std::nullopt);
    
    /**
     * @brief Calculate Stochastic Oscillator
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param kPeriod Period for the %K line
     * @param dPeriod Period for the %D line
     * @param kName Name for the %K column (default: "stoch_k")
     * @param dName Name for the %D column (default: "stoch_d")
     * @return std::shared_ptr<arrow::Table> Table with Stochastic Oscillator columns added
     */
    std::shared_ptr<arrow::Table> calculateStochastic(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int kPeriod = 14,
        int dPeriod = 3,
        std::optional<std::string> kName = std::nullopt,
        std::optional<std::string> dName = std::nullopt);
    
    /**
     * @brief Calculate Average Directional Index (ADX)
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param period Period for ADX calculation
     * @param adxName Name for the ADX column (default: "adx")
     * @param plusDIName Name for the +DI column (default: "plus_di")
     * @param minusDIName Name for the -DI column (default: "minus_di")
     * @return std::shared_ptr<arrow::Table> Table with ADX columns added
     */
    std::shared_ptr<arrow::Table> calculateADX(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int period = 14,
        std::optional<std::string> adxName = std::nullopt,
        std::optional<std::string> plusDIName = std::nullopt,
        std::optional<std::string> minusDIName = std::nullopt);
    
    /**
     * @brief Calculate On-Balance Volume (OBV)
     * 
     * @param table Input Arrow table
     * @param closeColumn Column name for the close prices
     * @param volumeColumn Column name for the volume
     * @param newColumnName Name for the new column (default: "obv")
     * @return std::shared_ptr<arrow::Table> Table with OBV column added
     */
    std::shared_ptr<arrow::Table> calculateOBV(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& closeColumn,
        const std::string& volumeColumn,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Ichimoku Cloud
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param conversionPeriod Period for the conversion line (default: 9)
     * @param basePeriod Period for the base line (default: 26)
     * @param laggingSpanPeriod Period for the lagging span (default: 52)
     * @param displacement Displacement period (default: 26)
     * @return std::shared_ptr<arrow::Table> Table with Ichimoku Cloud columns added
     */
    std::shared_ptr<arrow::Table> calculateIchimoku(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int conversionPeriod = 9,
        int basePeriod = 26,
        int laggingSpanPeriod = 52,
        int displacement = 26);
    
    /**
     * @brief Calculate Rate of Change (ROC)
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate ROC on
     * @param period Period for ROC calculation
     * @param newColumnName Name for the new column (default: "{column}_roc_{period}")
     * @return std::shared_ptr<arrow::Table> Table with ROC column added
     */
    std::shared_ptr<arrow::Table> calculateROC(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Average True Range (ATR)
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param period Period for ATR calculation
     * @param newColumnName Name for the new column (default: "atr_{period}")
     * @return std::shared_ptr<arrow::Table> Table with ATR column added
     */
    std::shared_ptr<arrow::Table> calculateATR(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int period = 14,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Commodity Channel Index (CCI)
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param period Period for CCI calculation
     * @param newColumnName Name for the new column (default: "cci_{period}")
     * @return std::shared_ptr<arrow::Table> Table with CCI column added
     */
    std::shared_ptr<arrow::Table> calculateCCI(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int period = 20,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Williams %R
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param period Period for Williams %R calculation
     * @param newColumnName Name for the new column (default: "williams_r_{period}")
     * @return std::shared_ptr<arrow::Table> Table with Williams %R column added
     */
    std::shared_ptr<arrow::Table> calculateWilliamsR(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int period = 14,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Parabolic SAR
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param accelerationFactor Initial acceleration factor
     * @param accelerationMax Maximum acceleration factor
     * @param newColumnName Name for the new column (default: "psar")
     * @return std::shared_ptr<arrow::Table> Table with Parabolic SAR column added
     */
    std::shared_ptr<arrow::Table> calculateParabolicSAR(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        double accelerationFactor = 0.02,
        double accelerationMax = 0.2,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Standard Deviation
     * 
     * @param table Input Arrow table
     * @param column Column name to calculate standard deviation on
     * @param period Period for standard deviation calculation
     * @param newColumnName Name for the new column (default: "{column}_stddev_{period}")
     * @return std::shared_ptr<arrow::Table> Table with standard deviation column added
     */
    std::shared_ptr<arrow::Table> calculateStdDev(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& column,
        int period,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Keltner Channels
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param emaPeriod Period for the EMA
     * @param atrPeriod Period for the ATR
     * @param multiplier Multiplier for the ATR
     * @param upperChannelName Name for the upper channel column (default: "keltner_upper")
     * @param middleChannelName Name for the middle channel column (default: "keltner_middle")
     * @param lowerChannelName Name for the lower channel column (default: "keltner_lower")
     * @return std::shared_ptr<arrow::Table> Table with Keltner Channels columns added
     */
    std::shared_ptr<arrow::Table> calculateKeltnerChannels(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        int emaPeriod = 20,
        int atrPeriod = 10,
        double multiplier = 2.0,
        std::optional<std::string> upperChannelName = std::nullopt,
        std::optional<std::string> middleChannelName = std::nullopt,
        std::optional<std::string> lowerChannelName = std::nullopt);
    
    /**
     * @brief Calculate Donchian Channels
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param period Period for Donchian Channels calculation
     * @param upperChannelName Name for the upper channel column (default: "donchian_upper")
     * @param middleChannelName Name for the middle channel column (default: "donchian_middle")
     * @param lowerChannelName Name for the lower channel column (default: "donchian_lower")
     * @return std::shared_ptr<arrow::Table> Table with Donchian Channels columns added
     */
    std::shared_ptr<arrow::Table> calculateDonchianChannels(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        int period = 20,
        std::optional<std::string> upperChannelName = std::nullopt,
        std::optional<std::string> middleChannelName = std::nullopt,
        std::optional<std::string> lowerChannelName = std::nullopt);
    
    /**
     * @brief Calculate Awesome Oscillator
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param shortPeriod Period for the short SMA (default: 5)
     * @param longPeriod Period for the long SMA (default: 34)
     * @param newColumnName Name for the new column (default: "awesome_oscillator")
     * @return std::shared_ptr<arrow::Table> Table with Awesome Oscillator column added
     */
    std::shared_ptr<arrow::Table> calculateAwesomeOscillator(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        int shortPeriod = 5,
        int longPeriod = 34,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Money Flow Index (MFI)
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param volumeColumn Column name for the volume
     * @param period Period for MFI calculation
     * @param newColumnName Name for the new column (default: "mfi_{period}")
     * @return std::shared_ptr<arrow::Table> Table with MFI column added
     */
    std::shared_ptr<arrow::Table> calculateMFI(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        const std::string& volumeColumn,
        int period = 14,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Chaikin Money Flow (CMF)
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param volumeColumn Column name for the volume
     * @param period Period for CMF calculation
     * @param newColumnName Name for the new column (default: "cmf_{period}")
     * @return std::shared_ptr<arrow::Table> Table with CMF column added
     */
    std::shared_ptr<arrow::Table> calculateCMF(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        const std::string& volumeColumn,
        int period = 20,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate Chaikin Oscillator
     * 
     * @param table Input Arrow table
     * @param highColumn Column name for the high prices
     * @param lowColumn Column name for the low prices
     * @param closeColumn Column name for the close prices
     * @param volumeColumn Column name for the volume
     * @param fastPeriod Period for the fast EMA (default: 3)
     * @param slowPeriod Period for the slow EMA (default: 10)
     * @param newColumnName Name for the new column (default: "chaikin_oscillator")
     * @return std::shared_ptr<arrow::Table> Table with Chaikin Oscillator column added
     */
    std::shared_ptr<arrow::Table> calculateChaikinOscillator(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& highColumn,
        const std::string& lowColumn,
        const std::string& closeColumn,
        const std::string& volumeColumn,
        int fastPeriod = 3,
        int slowPeriod = 10,
        std::optional<std::string> newColumnName = std::nullopt);
    
    /**
     * @brief Calculate multiple indicators at once
     * 
     * @param table Input Arrow table
     * @param indicators List of indicator names to calculate
     * @param params Parameters for each indicator
     * @return std::shared_ptr<arrow::Table> Table with all indicators added
     */
    std::shared_ptr<arrow::Table> calculateMultipleIndicators(
        const std::shared_ptr<arrow::Table>& table,
        const std::vector<std::string>& indicators,
        const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& params);
    
private:
    bool debug_ = false;
    
    // Helper functions for indicator calculations
    std::shared_ptr<arrow::Array> calculateMovingAverage(
        const std::shared_ptr<arrow::Array>& array,
        int period);
    
    std::shared_ptr<arrow::Array> calculateExponentialMovingAverage(
        const std::shared_ptr<arrow::Array>& array,
        int period);
    
    std::shared_ptr<arrow::Array> calculateTrueRange(
        const std::shared_ptr<arrow::Array>& highArray,
        const std::shared_ptr<arrow::Array>& lowArray,
        const std::shared_ptr<arrow::Array>& closeArray);
    
    // Helper function to handle edge cases
    std::shared_ptr<arrow::Array> handleEdgeCases(
        const std::shared_ptr<arrow::Array>& array,
        int period,
        double defaultValue = 0.0);
    
    // Helper function to add a column to a table
    std::shared_ptr<arrow::Table> addColumn(
        const std::shared_ptr<arrow::Table>& table,
        const std::string& name,
        const std::shared_ptr<arrow::Array>& array);
};

#endif // TECHNICAL_INDICATORS_H