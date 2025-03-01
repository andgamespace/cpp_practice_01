#include "frontendController.h"
#include <drogon/drogon.h>
#include <json/json.h>
#include <arrow/api.h>
#include <algorithm>
#include <sstream>

// Constructor: Loads time series data for a predefined set of tickers.
// In production, you might load data dynamically or update it in real time.
FrontendController::FrontendController()
{
    // List of tickers to load.
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};
    // Adjust the base directory as needed.
    std::string baseDir = "../src/stock_data/";

    for (const auto &ticker : tickers)
    {
        std::vector<std::string> filePaths;
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min.csv");
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min(1).csv");
        filePaths.push_back(baseDir + "time-series-" + ticker + "-5min(2).csv");
        timeSeriesLoader_.loadTickerData(ticker, filePaths);
    }
}

// Helper function: Converts an Apache Arrow Table into a JSON array.
Json::Value FrontendController::arrowTableToJson(const std::shared_ptr<arrow::Table> &table)
{
    Json::Value jsonData(Json::arrayValue);

    if (!table) return jsonData;

    // Assume columns: datetime, open, high, low, close, volume.
    auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
    auto openArray     = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    auto highArray     = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
    auto lowArray      = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
    auto closeArray    = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
    auto volumeArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));

    int numRows = table->num_rows();
    for (int i = 0; i < numRows; ++i)
    {
        Json::Value row;
        row["datetime"] = datetimeArray->GetString(i);
        row["open"]     = openArray->Value(i);
        row["high"]     = highArray->Value(i);
        row["low"]      = lowArray->Value(i);
        row["close"]    = closeArray->Value(i);
        row["volume"]   = volumeArray->Value(i);
        jsonData.append(row);
    }
    return jsonData;
}

// GET /time-series/{ticker}
// Retrieves time series data for the specified ticker and sends it as JSON.
void FrontendController::getTimeSeriesData(const HttpRequestPtr &req,
                                           std::function<void (const HttpResponsePtr &)> &&callback,
                                           const std::string &ticker)
{
    auto table = timeSeriesLoader_.getTickerData(ticker);
    Json::Value responseJson;
    if (!table)
    {
        responseJson["error"] = "Time series data not found for ticker: " + ticker;
    }
    else
    {
        responseJson["ticker"] = ticker;
        responseJson["data"] = arrowTableToJson(table);
    }
    auto resp = HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// GET /portfolio/live
// Returns live portfolio metrics.
// In a real system, these metrics would be updated in real time.
void FrontendController::getLivePortfolioMetrics(const HttpRequestPtr &req,
                                                 std::function<void (const HttpResponsePtr &)> &&callback)
{
    Json::Value responseJson;
    responseJson["portfolio"] = getMockPortfolioMetrics();
    auto resp = HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// GET /portfolio/winloss
// Returns the win/loss ratio for the portfolio.
void FrontendController::getWinLossRatio(const HttpRequestPtr &req,
                                         std::function<void (const HttpResponsePtr &)> &&callback)
{
    Json::Value responseJson;
    responseJson["winLossRatio"] = getMockWinLossRatio();
    auto resp = HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// GET /portfolio/pnl
// Returns profit and loss metrics for the portfolio.
void FrontendController::getProfitLoss(const HttpRequestPtr &req,
                                       std::function<void (const HttpResponsePtr &)> &&callback)
{
    Json::Value responseJson;
    responseJson["profitLoss"] = getMockProfitLoss();
    auto resp = HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// Mock helper: Returns sample live portfolio metrics.
Json::Value FrontendController::getMockPortfolioMetrics()
{
    Json::Value metrics;
    metrics["timestamp"] = "2025-02-28 15:30:00";
    metrics["balance"] = 100000.0;
    metrics["equity"] = 105000.0;
    metrics["openPositions"] = 3;
    metrics["profitLoss"] = 5000.0;
    return metrics;
}

// Mock helper: Returns a sample win/loss ratio.
Json::Value FrontendController::getMockWinLossRatio()
{
    Json::Value ratio;
    ratio["wins"] = 12;
    ratio["losses"] = 8;
    ratio["winLossRatio"] = 1.5;
    return ratio;
}

// Mock helper: Returns a sample profit and loss breakdown.
Json::Value FrontendController::getMockProfitLoss()
{
    Json::Value pnl;
    pnl["totalPnL"] = 5000.0;
    pnl["dailyPnL"] = Json::arrayValue;
    // For illustration, populate some daily PnL data.
    for (int i = 0; i < 5; ++i)
    {
        Json::Value day;
        day["date"] = "2025-02-2" + std::to_string(i+5);
        day["pnl"] = 1000.0 - i * 200;  // sample values
        pnl["dailyPnL"].append(day);
    }
    return pnl;
}
