#ifndef FRONTENDCONTROLLER_H
#define FRONTENDCONTROLLER_H
#pragma once

#include <drogon/HttpController.h>
#include <json/json.h>
#include <memory>
#include "DataLoader.h"
#include "BacktestEngine.h"

/**
 * @brief FrontendController handles REST endpoints to serve time-series
 *        data and portfolio metrics for a React (or other) frontend.
 */
class FrontendController : public drogon::HttpController<FrontendController>
{
public:
    FrontendController();
    
    // Set the BacktestEngine to use for real metrics
    void setBacktestEngine(std::shared_ptr<BacktestEngine> engine);

    // GET /time-series/{ticker}
    void getTimeSeriesData(const drogon::HttpRequestPtr &req,
                           std::function<void(const drogon::HttpResponsePtr &)> &&callback,
                           const std::string &ticker);

    // GET /portfolio/live
    void getLivePortfolioMetrics(const drogon::HttpRequestPtr &req,
                                 std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    // GET /portfolio/winloss
    void getWinLossRatio(const drogon::HttpRequestPtr &req,
                         std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    // GET /portfolio/pnl
    void getProfitLoss(const drogon::HttpRequestPtr &req,
                       std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    // GET /portfolio/supervised
    void getSupervisedLearningMetrics(const drogon::HttpRequestPtr &req,
                                      std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    // GET /portfolio/rl
    void getReinforcementLearningMetrics(const drogon::HttpRequestPtr &req,
                                         std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    // GET /portfolio/performance
    void getPerformanceOverview(const drogon::HttpRequestPtr &req,
                                std::function<void(const drogon::HttpResponsePtr &)> &&callback);
                                
    // GET /tickers
    void getAvailableTickers(const drogon::HttpRequestPtr &req,
                             std::function<void(const drogon::HttpResponsePtr &)> &&callback);
                             
    // GET /transactions
    void getTransactions(const drogon::HttpRequestPtr &req,
                         std::function<void(const drogon::HttpResponsePtr &)> &&callback);
                         
    // POST /backtest/start
    void startBacktest(const drogon::HttpRequestPtr &req,
                       std::function<void(const drogon::HttpResponsePtr &)> &&callback);
                       
    // POST /backtest/reset
    void resetBacktest(const drogon::HttpRequestPtr &req,
                       std::function<void(const drogon::HttpResponsePtr &)> &&callback);

    METHOD_LIST_BEGIN
        ADD_METHOD_TO(FrontendController::getTimeSeriesData, "/time-series/{1}", drogon::Get);
        ADD_METHOD_TO(FrontendController::getLivePortfolioMetrics, "/portfolio/live", drogon::Get);
        ADD_METHOD_TO(FrontendController::getWinLossRatio, "/portfolio/winloss", drogon::Get);
        ADD_METHOD_TO(FrontendController::getProfitLoss, "/portfolio/pnl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getSupervisedLearningMetrics, "/portfolio/supervised", drogon::Get);
        ADD_METHOD_TO(FrontendController::getReinforcementLearningMetrics, "/portfolio/rl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getPerformanceOverview, "/portfolio/performance", drogon::Get);
        ADD_METHOD_TO(FrontendController::getAvailableTickers, "/tickers", drogon::Get);
        ADD_METHOD_TO(FrontendController::getTransactions, "/transactions", drogon::Get);
        ADD_METHOD_TO(FrontendController::startBacktest, "/backtest/start", drogon::Post);
        ADD_METHOD_TO(FrontendController::resetBacktest, "/backtest/reset", drogon::Post);
    METHOD_LIST_END

private:
    DataLoader timeSeriesLoader_;
    std::shared_ptr<BacktestEngine> engine_;
    bool debug_ = false;
    std::mutex mtx_;

    // Convert an Arrow table to JSON for the frontend.
    Json::Value arrowTableToJson(const std::shared_ptr<arrow::Table> &table);

    // Get real metrics from the engine if available, otherwise use mock data
    Json::Value getPortfolioMetrics();
    Json::Value getWinLossRatioData();
    Json::Value getProfitLossData();
    Json::Value getPerformanceOverviewData();
    
    // Convert a transaction to JSON
    Json::Value transactionToJson(const BacktestEngine::Transaction &tx);

    // Mock metrics (used as fallback when no engine is available)
    Json::Value getMockPortfolioMetrics();
    Json::Value getMockWinLossRatio();
    Json::Value getMockProfitLoss();
    Json::Value getMockPerformanceOverview();
};

#endif // FRONTENDCONTROLLER_H
