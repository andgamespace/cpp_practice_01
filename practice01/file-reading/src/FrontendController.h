#ifndef FRONTENDCONTROLLER_H
#define FRONTENDCONTROLLER_H
#pragma once

#include <drogon/HttpController.h>  // Required for HttpController, ADD_METHOD_TO, etc.
#include <json/json.h>
#include "DataLoader.h"

/**
 * @brief FrontendController is responsible for sending time series and portfolio metrics data
 *        to the React (or any) frontend. It abstracts functionality for:
 *        - Time series data
 *        - Live portfolio metrics
 *        - Win/loss ratio
 *        - Profit and loss
 *        - Supervised/Deep learning metrics
 *        - Reinforcement learning metrics
 *        - Combined performance overview (win/loss, PnL direction, average win/loss, etc.)
 */
class FrontendController : public drogon::HttpController<FrontendController>
{
public:
    FrontendController();

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

    METHOD_LIST_BEGIN
        // Explicitly use drogon::Get instead of just Get:
        ADD_METHOD_TO(FrontendController::getTimeSeriesData, "/time-series/{1}", drogon::Get);
        ADD_METHOD_TO(FrontendController::getLivePortfolioMetrics, "/portfolio/live", drogon::Get);
        ADD_METHOD_TO(FrontendController::getWinLossRatio,         "/portfolio/winloss", drogon::Get);
        ADD_METHOD_TO(FrontendController::getProfitLoss,           "/portfolio/pnl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getSupervisedLearningMetrics, "/portfolio/supervised", drogon::Get);
        ADD_METHOD_TO(FrontendController::getReinforcementLearningMetrics, "/portfolio/rl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getPerformanceOverview,  "/portfolio/performance", drogon::Get);
    METHOD_LIST_END

private:
    // DataLoader instance to load time series CSV data
    DataLoader timeSeriesLoader_;
    bool debug_ = true;

    // Convert an Apache Arrow Table to JSON
    Json::Value arrowTableToJson(const std::shared_ptr<arrow::Table> &table);

    // Mock or real data retrieval for portfolio stats
    Json::Value getMockPortfolioMetrics();
    Json::Value getMockWinLossRatio();
    Json::Value getMockProfitLoss();

    // Combined mock aggregator
    Json::Value getMockPerformanceOverview();
};

#endif // FRONTENDCONTROLLER_H
