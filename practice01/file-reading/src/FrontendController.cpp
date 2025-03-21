#include "FrontendController.h"
#include <drogon/drogon.h>
#include <spdlog/spdlog.h>
#include <arrow/api.h>
#include <json/json.h>
#include <chrono>
#include <thread>
#include <future>

FrontendController::FrontendController() : debug_(false), engine_(nullptr)
{
    if (debug_) spdlog::info("Initializing FrontendController");
    
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
            if (debug_) spdlog::error("Failed to load data for ticker: {}", ticker);
        } else {
            if (debug_) spdlog::info("Loaded data for ticker: {}", ticker);
        }
    }
}

void FrontendController::setBacktestEngine(std::shared_ptr<BacktestEngine> engine) {
    std::lock_guard<std::mutex> lock(mtx_);
    engine_ = engine;
    if (debug_) spdlog::info("BacktestEngine set in FrontendController");
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
    if (debug_) spdlog::info("Request: time series data for ticker: {}", ticker);
    
    auto table = timeSeriesLoader_.getTickerData(ticker);
    Json::Value responseJson;
    
    if (!table) {
        responseJson["error"] = "Time series data not found for ticker: " + ticker;
        if (debug_) spdlog::error("Data not found for ticker: {}", ticker);
    } else {
        responseJson["ticker"] = ticker;
        responseJson["data"] = arrowTableToJson(table);
    }
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

Json::Value FrontendController::getPortfolioMetrics() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (engine_) {
        Json::Value metrics;
        Json::Reader reader;
        
        // Parse the JSON string from the engine
        std::string jsonStr = engine_->getPortfolioMetricsJson();
        if (reader.parse(jsonStr, metrics)) {
            return metrics;
        }
    }
    
    // Fallback to mock data if engine is not available or parsing fails
    return getMockPortfolioMetrics();
}

Json::Value FrontendController::getWinLossRatioData() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (engine_) {
        auto metrics = engine_->getPerformanceMetrics();
        
        Json::Value ratio;
        ratio["wins"] = metrics.winningTrades;
        ratio["losses"] = metrics.losingTrades;
        ratio["winLossRatio"] = metrics.winRate > 0 ? metrics.winRate : 0;
        
        return ratio;
    }
    
    // Fallback to mock data
    return getMockWinLossRatio();
}

Json::Value FrontendController::getProfitLossData() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (engine_) {
        auto metrics = engine_->getPerformanceMetrics();
        
        Json::Value pnl;
        pnl["totalPnL"] = (metrics.finalBalance - metrics.initialBalance);
        pnl["percentReturn"] = metrics.totalReturn * 100.0;
        pnl["annualizedReturn"] = metrics.annualizedReturn * 100.0;
        
        // Create mock daily PnL for now
        // In a real implementation, you would track this over time
        pnl["dailyPnL"] = Json::arrayValue;
        for (int i = 0; i < 5; ++i) {
            Json::Value day;
            day["date"] = "2025-03-" + std::to_string(10 + i);
            day["pnl"] = (metrics.finalBalance - metrics.initialBalance) / 5.0;
            pnl["dailyPnL"].append(day);
        }
        
        return pnl;
    }
    
    // Fallback to mock data
    return getMockProfitLoss();
}

Json::Value FrontendController::getPerformanceOverviewData() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (engine_) {
        auto metrics = engine_->getPerformanceMetrics();
        auto winLoss = getWinLossRatioData();
        auto pnl = getProfitLossData();
        
        Json::Value overview;
        overview["timestamp"] = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        overview["totalTrades"] = metrics.totalTrades;
        overview["wins"] = metrics.winningTrades;
        overview["losses"] = metrics.losingTrades;
        overview["winLossRatio"] = metrics.winRate;
        overview["totalPnL"] = metrics.finalBalance - metrics.initialBalance;
        overview["avgWin"] = metrics.averageWin;
        overview["avgLoss"] = metrics.averageLoss;
        overview["pnlDirection"] = (metrics.totalReturn >= 0) ? "GREEN" : "RED";
        overview["pnlMagnitude%"] = metrics.totalReturn * 100.0;
        overview["dailyPnL"] = pnl["dailyPnL"];
        overview["sharpeRatio"] = metrics.sharpeRatio;
        overview["maxDrawdown"] = metrics.maxDrawdown;
        
        Json::Value rl;
        rl["episodes"] = 1;
        rl["avgReward"] = metrics.totalReturn;
        rl["expectancy"] = metrics.expectancy;
        overview["rlStats"] = rl;
        
        return overview;
    }
    
    // Fallback to mock data
    return getMockPerformanceOverview();
}

Json::Value FrontendController::transactionToJson(const BacktestEngine::Transaction &tx) {
    Json::Value json;
    
    switch (tx.action) {
        case BacktestEngine::Action::Buy:
            json["action"] = "BUY";
            break;
        case BacktestEngine::Action::Sell:
            json["action"] = "SELL";
            break;
        case BacktestEngine::Action::Hold:
            json["action"] = "HOLD";
            break;
    }
    
    json["ticker"] = tx.ticker;
    json["quantity"] = tx.quantity;
    json["price"] = tx.price;
    json["datetime"] = tx.datetime;
    json["value"] = tx.price * tx.quantity;
    
    return json;
}

void FrontendController::getLivePortfolioMetrics(const drogon::HttpRequestPtr &req,
                                                 std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: live portfolio metrics");
    
    Json::Value responseJson;
    responseJson["portfolio"] = getPortfolioMetrics();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getWinLossRatio(const drogon::HttpRequestPtr &req,
                                         std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: win/loss ratio");
    
    Json::Value responseJson;
    responseJson["winLossRatio"] = getWinLossRatioData();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getProfitLoss(const drogon::HttpRequestPtr &req,
                                       std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: profit/loss metrics");
    
    Json::Value responseJson;
    responseJson["profitLoss"] = getProfitLossData();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getSupervisedLearningMetrics(const drogon::HttpRequestPtr &req,
                                                      std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: supervised learning metrics");
    
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
    if (debug_) spdlog::info("Request: reinforcement learning metrics");
    
    Json::Value responseJson;
    
    if (engine_) {
        auto metrics = engine_->getPerformanceMetrics();
        
        responseJson["episodes"] = 1;
        responseJson["totalReward"] = metrics.totalReturn;
        responseJson["averageReward"] = metrics.totalReturn;
        
        // Example placeholders
        Json::Value tensor;
        tensor["policyGradient"] = metrics.winRate;
        tensor["qValue"] = metrics.expectancy;
        responseJson["tensors"] = tensor;
    } else {
        responseJson["episodes"] = 100;
        responseJson["totalReward"] = 2500;
        responseJson["averageReward"] = 25;
        
        Json::Value tensor;
        tensor["policyGradient"] = 0.15;
        tensor["qValue"] = 1.23;
        responseJson["tensors"] = tensor;
    }
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getPerformanceOverview(const drogon::HttpRequestPtr &req,
                                                std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: combined performance overview");
    
    Json::Value responseJson = getPerformanceOverviewData();
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getAvailableTickers(const drogon::HttpRequestPtr &req,
                                             std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: available tickers");
    
    Json::Value responseJson;
    Json::Value tickersArray(Json::arrayValue);
    
    std::vector<std::string> tickers;
    if (engine_) {
        tickers = engine_->getAvailableTickers();
    } else {
        // Fallback to data loader
        tickers = timeSeriesLoader_.getAvailableTickers();
    }
    
    for (const auto &ticker : tickers) {
        tickersArray.append(ticker);
    }
    
    responseJson["tickers"] = tickersArray;
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::getTransactions(const drogon::HttpRequestPtr &req,
                                         std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: transactions");
    
    Json::Value responseJson;
    Json::Value transactionsArray(Json::arrayValue);
    
    if (engine_) {
        const auto &transactions = engine_->getTransactions();
        
        for (const auto &tx : transactions) {
            transactionsArray.append(transactionToJson(tx));
        }
    }
    
    responseJson["transactions"] = transactionsArray;
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::startBacktest(const drogon::HttpRequestPtr &req,
                                       std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: start backtest");
    
    Json::Value responseJson;
    
    if (engine_) {
        // Start backtest in a separate thread
        std::thread([this]() {
            this->engine_->runBacktest();
        }).detach();
        
        responseJson["status"] = "success";
        responseJson["message"] = "Backtest started";
    } else {
        responseJson["status"] = "error";
        responseJson["message"] = "No backtest engine available";
    }
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

void FrontendController::resetBacktest(const drogon::HttpRequestPtr &req,
                                       std::function<void(const drogon::HttpResponsePtr &)> &&callback)
{
    if (debug_) spdlog::info("Request: reset backtest");
    
    Json::Value responseJson;
    
    if (engine_) {
        engine_->reset();
        responseJson["status"] = "success";
        responseJson["message"] = "Backtest reset";
    } else {
        responseJson["status"] = "error";
        responseJson["message"] = "No backtest engine available";
    }
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(responseJson);
    callback(resp);
}

// Mock data (used as fallback when no engine is available)
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
    pnl["percentReturn"] = 4.5;
    pnl["annualizedReturn"] = 12.8;
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
    overview["sharpeRatio"] = 1.2;
    overview["maxDrawdown"] = 0.05;

    Json::Value rl;
    rl["episodes"] = 100;
    rl["avgReward"] = 25.0;
    rl["expectancy"] = 0.75;
    overview["rlStats"] = rl;

    return overview;
}
