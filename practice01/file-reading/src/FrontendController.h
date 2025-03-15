#ifndef FRONTENDCONTROLLER_H
#define FRONTENDCONTROLLER_H
#pragma once

#include <drogon/HttpController.h>
#include <json/json.h>
#include "DataLoader.h"

/**
 * @brief FrontendController handles REST endpoints to serve time-series
 *        data and mock portfolio metrics for a React (or other) frontend.
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
        ADD_METHOD_TO(FrontendController::getTimeSeriesData, "/time-series/{1}", drogon::Get);
        ADD_METHOD_TO(FrontendController::getLivePortfolioMetrics, "/portfolio/live", drogon::Get);
        ADD_METHOD_TO(FrontendController::getWinLossRatio, "/portfolio/winloss", drogon::Get);
        ADD_METHOD_TO(FrontendController::getProfitLoss, "/portfolio/pnl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getSupervisedLearningMetrics, "/portfolio/supervised", drogon::Get);
        ADD_METHOD_TO(FrontendController::getReinforcementLearningMetrics, "/portfolio/rl", drogon::Get);
        ADD_METHOD_TO(FrontendController::getPerformanceOverview, "/portfolio/performance", drogon::Get);
    METHOD_LIST_END

private:
    DataLoader timeSeriesLoader_;
    bool debug_ = true;

    // Convert an Arrow table to JSON for the frontend.
    Json::Value arrowTableToJson(const std::shared_ptr<arrow::Table> &table);

    // Mock metrics
    Json::Value getMockPortfolioMetrics();
    Json::Value getMockWinLossRatio();
    Json::Value getMockProfitLoss();
    Json::Value getMockPerformanceOverview();
};

#endif // FRONTENDCONTROLLER_H
