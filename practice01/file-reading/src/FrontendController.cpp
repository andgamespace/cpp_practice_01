#include "FrontendController.h"
#include <drogon/drogon.h>
#include <spdlog/spdlog.h>
#include <arrow/api.h>
#include <json/json.h>

FrontendController::FrontendController() : debug_(true)
{
    spdlog::info("Initializing FrontendController");
    // For demonstration, we load a few tickers automatically
    std::vector<std::string> tickers = {"AAPL", "MSFT", "NVDA", "AMD"};
    std::string baseDir = "../src/stock_data/";
    for (const auto &ticker : tickers) {
        std::vector<std::string> filePaths = {
            baseDir + "time-series-" + ticker + "-5min.csv",
            baseDir + "time-series-" + ticker + "-5min(1).csv",
            baseDir + "time-series-" + ticker + "-5min(2).csv"
        };
        if (!timeSeriesLoader_.loadTickerData(ticker, filePaths)) {
            spdlog::error("Failed to load data for ticker: {}", ticker);
        } else {
            spdlog::info("Loaded data for ticker: {}", ticker);
        }
    }
}

Json::Value FrontendController::arrowTableToJson(const std::shared_ptr<arrow::Table> &table)
{
    Json::Value jsonData(Json::arrayValue);
    if (!table) return jsonData;

    // [datetime, open, high, low, close, volume]
    auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
    auto openArray     = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    auto highArray     = std::static_pointer_cast<arrow::DoubleArray>(table->column(2)->chunk(0));
    auto lowArray      = std::static_pointer_cast<arrow::DoubleArray>(table->column(3)->chunk(0));
    auto closeArray    = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
    auto volumeArray   = std::static_pointer_cast<arrow::DoubleArray>(table->column(5)->chunk(0));

    int numRows = table->num_rows();
    for (int i = 0; i < numRows; ++i) {
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

void FrontendController::getTimeSeriesData(const drogon::HttpRequestPtr &req,
                                           std::function<void(const drogon::HttpResponsePtr &)> &&callback,
                                           const std::string &ticker)
{
    spdlog::info("Request: time series data for ticker: {}", ticker);
    auto table = timeSeriesLoader_.getTickerData(ticker);
    Json::Value responseJson;
    if (!table) {
        responseJson["error"] = "Time series data not found for ticker: " + ticker;
        spdlog::error("Data not found for ticker: {}", ticker);
    } else {
        responseJson["ticker"] = ticker;
        responseJson["data"] = arrowTableToJson(table);
    }
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// Below: these endpoints return mock data, but in production
// you might connect them to your real portfolio state, etc.

void FrontendController::getLivePortfolioMetrics(const drogon::HttpRequestPtr &req,
                                                 std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: live portfolio metrics");
    Json::Value responseJson;
    responseJson["portfolio"] = getMockPortfolioMetrics();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getWinLossRatio(const drogon::HttpRequestPtr &req,
                                         std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: win/loss ratio");
    Json::Value responseJson;
    responseJson["winLossRatio"] = getMockWinLossRatio();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getProfitLoss(const drogon::HttpRequestPtr &req,
                                       std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: profit/loss metrics");
    Json::Value responseJson;
    responseJson["profitLoss"] = getMockProfitLoss();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getSupervisedLearningMetrics(const drogon::HttpRequestPtr &req,
                                                      std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: supervised learning metrics (mock)");
    Json::Value responseJson;
    responseJson["accuracy"] = 0.92;
    responseJson["loss"] = 0.08;
    responseJson["epochs"] = 50;
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getReinforcementLearningMetrics(const drogon::HttpRequestPtr &req,
                                                         std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: reinforcement learning metrics (mock)");
    Json::Value responseJson;
    responseJson["episodes"] = 100;
    responseJson["totalReward"] = 2500;
    responseJson["averageReward"] = 25;

    // Example placeholders
    Json::Value tensor;
    tensor["policyGradient"] = 0.15;
    tensor["qValue"] = 1.23;
    responseJson["tensors"] = tensor;

    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getPerformanceOverview(const drogon::HttpRequestPtr &req,
    std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    spdlog::info("Request: combined performance overview");
    Json::Value responseJson = getMockPerformanceOverview();
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// Mock data
Json::Value FrontendController::getMockPortfolioMetrics() {
    Json::Value metrics;
    metrics["timestamp"] = "2025-03-11 15:30:00";
    metrics["balance"] = 100000.0;
    metrics["equity"] = 104500.0;
    metrics["openPositions"] = 3;
    metrics["profitLoss"] = 4500.0;
    return metrics;
}

Json::Value FrontendController::getMockWinLossRatio() {
    Json::Value ratio;
    ratio["wins"] = 18;
    ratio["losses"] = 12;
    ratio["winLossRatio"] = 1.5;
    return ratio;
}

Json::Value FrontendController::getMockProfitLoss() {
    Json::Value pnl;
    pnl["totalPnL"] = 4500.0;
    pnl["dailyPnL"] = Json::arrayValue;
    for (int i = 0; i < 5; ++i) {
        Json::Value day;
        day["date"] = "2025-03-" + std::to_string(10 + i);
        day["pnl"] = 900.0 - (i * 100.0);
        pnl["dailyPnL"].append(day);
    }
    return pnl;
}

Json::Value FrontendController::getMockPerformanceOverview() {
    auto winLoss = getMockWinLossRatio();
    auto pnl = getMockProfitLoss();
    int wins = winLoss["wins"].asInt();
    int losses = winLoss["losses"].asInt();
    int total = wins + losses;
    double totalPnL = pnl["totalPnL"].asDouble();
    double avgWin = 250.0;
    double avgLoss = -150.0;
    std::string direction = (totalPnL >= 0) ? "GREEN" : "RED";
    double magnitudePct = (totalPnL / 100000.0) * 100.0;

    Json::Value overview;
    overview["timestamp"] = "2025-03-11 15:30:00";
    overview["totalTrades"] = total;
    overview["wins"] = wins;
    overview["losses"] = losses;
    overview["winLossRatio"] = winLoss["winLossRatio"];
    overview["totalPnL"] = totalPnL;
    overview["avgWin"] = avgWin;
    overview["avgLoss"] = avgLoss;
    overview["pnlDirection"] = direction;
    overview["pnlMagnitude%"] = magnitudePct;
    overview["dailyPnL"] = pnl["dailyPnL"];

    Json::Value rl;
    rl["episodes"] = 100;
    rl["avgReward"] = 25.0;
    rl["qValue"] = 1.23;
    overview["rlStats"] = rl;

    return overview;
}
