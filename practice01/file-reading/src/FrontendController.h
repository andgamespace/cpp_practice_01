#ifndef FRONTENDCONTROLLER_H
#define FRONTENDCONTROLLER_H
#pragma once

#include <drogon/HttpController.h>
#include "DataLoader.h"
#include <json/json.h>

using namespace drogon;

/**
 * @brief FrontendController is responsible for sending time series and portfolio metrics data
 *        to the React frontend. It abstracts functionality for live metrics, time series,
 *        win/loss ratio, and profit and loss.
 */
class FrontendController : public drogon::HttpController<FrontendController>
{
public:
    FrontendController();

    /**
     * @brief Get time series data for a specific ticker.
     * @param req The incoming HTTP request.
     * @param callback Function to send the HTTP response.
     * @param ticker The ticker symbol for which time series data is requested.
     */
    void getTimeSeriesData(const HttpRequestPtr &req,
                           std::function<void (const HttpResponsePtr &)> &&callback,
                           const std::string &ticker);

    /**
     * @brief Get live portfolio metrics.
     * @param req The incoming HTTP request.
     * @param callback Function to send the HTTP response.
     */
    void getLivePortfolioMetrics(const HttpRequestPtr &req,
                                 std::function<void (const HttpResponsePtr &)> &&callback);

    /**
     * @brief Get the win/loss ratio.
     * @param req The incoming HTTP request.
     * @param callback Function to send the HTTP response.
     */
    void getWinLossRatio(const HttpRequestPtr &req,
                         std::function<void (const HttpResponsePtr &)> &&callback);

    /**
     * @brief Get the profit and loss metrics.
     * @param req The incoming HTTP request.
     * @param callback Function to send the HTTP response.
     */
    void getProfitLoss(const HttpRequestPtr &req,
                       std::function<void (const HttpResponsePtr &)> &&callback);

    METHOD_LIST_BEGIN
        // GET /time-series/{ticker} returns time series data for the ticker.
        ADD_METHOD_TO(FrontendController::getTimeSeriesData, "/time-series/{1}", Get);
        // GET /portfolio/live returns live portfolio metrics.
        ADD_METHOD_TO(FrontendController::getLivePortfolioMetrics, "/portfolio/live", Get);
        // GET /portfolio/winloss returns the win/loss ratio.
        ADD_METHOD_TO(FrontendController::getWinLossRatio, "/portfolio/winloss", Get);
        // GET /portfolio/pnl returns profit and loss data.
        ADD_METHOD_TO(FrontendController::getProfitLoss, "/portfolio/pnl", Get);
    METHOD_LIST_END

private:
    // Instance of DataLoader used to load time series CSV data.
    DataLoader timeSeriesLoader_;

    // Helper functions that provide (mock) portfolio data.
    Json::Value getMockPortfolioMetrics();
    Json::Value getMockWinLossRatio();
    Json::Value getMockProfitLoss();

    // Helper: Converts an Apache Arrow Table into a JSON array.
    Json::Value arrowTableToJson(const std::shared_ptr<arrow::Table> &table);
};

#endif // FRONTENDCONTROLLER_H
