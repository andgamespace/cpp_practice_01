#include "MyWebSocketController.h"
#include <drogon/drogon.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

using namespace drogon;

MyWebSocketController::MyWebSocketController()
    : debug_(false), engine_(nullptr), periodicUpdatesRunning_(false)
{
    if (debug_) spdlog::info("Initializing MyWebSocketController");
}

void MyWebSocketController::setBacktestEngine(std::shared_ptr<BacktestEngine> engine) {
    engine_ = engine;
    if (debug_) spdlog::info("BacktestEngine set in MyWebSocketController");
}

void MyWebSocketController::handleNewConnection(const HttpRequestPtr &req,
                                                const WebSocketConnectionPtr &conn)
{
    if (debug_) spdlog::info("[WebSocket] New client connected");
    {
        std::lock_guard<std::mutex> lock(connMutex_);
        connections_.insert(conn);
    }
    
    // Send a welcome message
    Json::Value initMsg;
    initMsg["type"] = "welcome";
    initMsg["payload"] = "Connected to Trading Environment WebSocket";
    
    // Add available streams
    Json::Value streams(Json::arrayValue);
    streams.append("portfolio");
    streams.append("transactions");
    streams.append("performance");
    initMsg["available_streams"] = streams;
    
    conn->send(initMsg.toStyledString());
}

void MyWebSocketController::handleNewMessage(const WebSocketConnectionPtr &conn,
                                             std::string &&message,
                                             const WebSocketMessageType &type)
{
    if (type == WebSocketMessageType::Text) {
        if (debug_) spdlog::info("[WebSocket] Received message: {}", message);
        
        Json::Reader reader;
        Json::Value incoming;
        
        if (reader.parse(message, incoming)) {
            // Check if it's a command
            if (incoming.isMember("command")) {
                processCommand(conn, incoming);
            } else {
                // Echo back for other messages
                Json::Value response;
                response["type"] = "echo";
                response["payload"] = incoming;
                conn->send(response.toStyledString());
            }
        } else {
            // Invalid JSON
            Json::Value errorMsg;
            errorMsg["type"] = "error";
            errorMsg["message"] = "Invalid JSON format";
            conn->send(errorMsg.toStyledString());
        }
    }
}

void MyWebSocketController::handleConnectionClosed(const WebSocketConnectionPtr &conn)
{
    if (debug_) spdlog::info("[WebSocket] Client disconnected");
    
    {
        std::lock_guard<std::mutex> lock(connMutex_);
        connections_.erase(conn);
    }
    
    // Remove from all subscriptions
    {
        std::lock_guard<std::mutex> lock(subMutex_);
        for (auto &pair : subscriptions_) {
            pair.second.erase(conn);
        }
    }
}

void MyWebSocketController::broadcastJson(const Json::Value &data)
{
    const std::string out = data.toStyledString();
    std::lock_guard<std::mutex> lock(connMutex_);
    
    for (auto &c : connections_) {
        if (c && c->connected()) {
            c->send(out);
        }
    }
}

void MyWebSocketController::broadcastMessage(const std::string &message)
{
    std::lock_guard<std::mutex> lock(connMutex_);
    
    for (auto &c : connections_) {
        if (c && c->connected()) {
            c->send(message);
        }
    }
}

void MyWebSocketController::broadcastPortfolioMetrics()
{
    if (!engine_) return;
    
    Json::Value data;
    data["type"] = "portfolio_update";
    
    Json::Reader reader;
    Json::Value metrics;
    
    // Parse the JSON string from the engine
    std::string jsonStr = engine_->getPortfolioMetricsJson();
    if (reader.parse(jsonStr, metrics)) {
        data["data"] = metrics;
        
        // Broadcast to all subscribers of the portfolio stream
        std::lock_guard<std::mutex> lock(subMutex_);
        auto it = subscriptions_.find("portfolio");
        
        if (it != subscriptions_.end()) {
            const std::string out = data.toStyledString();
            
            for (auto &conn : it->second) {
                if (conn && conn->connected()) {
                    conn->send(out);
                }
            }
        }
    }
}

void MyWebSocketController::startPeriodicUpdates(int intervalMs)
{
    std::lock_guard<std::mutex> lock(periodicMutex_);
    
    if (periodicUpdatesRunning_) {
        if (debug_) spdlog::warn("Periodic updates already running");
        return;
    }
    
    periodicUpdatesRunning_ = true;
    
    // Start a thread for periodic updates
    periodicThread_ = std::thread([this, intervalMs]() {
        while (periodicUpdatesRunning_) {
            // Broadcast portfolio metrics
            this->broadcastPortfolioMetrics();
            
            // Sleep for the specified interval
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
        }
    });
    
    // Detach the thread so it runs independently
    periodicThread_.detach();
    
    if (debug_) spdlog::info("Started periodic updates with interval {} ms", intervalMs);
}

void MyWebSocketController::stopPeriodicUpdates()
{
    std::lock_guard<std::mutex> lock(periodicMutex_);
    
    if (!periodicUpdatesRunning_) {
        if (debug_) spdlog::warn("Periodic updates not running");
        return;
    }
    
    periodicUpdatesRunning_ = false;
    
    if (debug_) spdlog::info("Stopped periodic updates");
}

void MyWebSocketController::processCommand(const WebSocketConnectionPtr &conn, const Json::Value &command)
{
    std::string cmd = command["command"].asString();
    
    if (cmd == "subscribe") {
        if (command.isMember("stream")) {
            std::string stream = command["stream"].asString();
            subscribeClient(conn, stream);
            
            // Send confirmation
            Json::Value response;
            response["type"] = "subscription";
            response["status"] = "success";
            response["stream"] = stream;
            conn->send(response.toStyledString());
        }
    }
    else if (cmd == "unsubscribe") {
        if (command.isMember("stream")) {
            std::string stream = command["stream"].asString();
            unsubscribeClient(conn, stream);
            
            // Send confirmation
            Json::Value response;
            response["type"] = "unsubscription";
            response["status"] = "success";
            response["stream"] = stream;
            conn->send(response.toStyledString());
        }
    }
    else if (cmd == "get_portfolio") {
        if (engine_) {
            Json::Value response;
            response["type"] = "portfolio";
            
            Json::Reader reader;
            Json::Value metrics;
            
            // Parse the JSON string from the engine
            std::string jsonStr = engine_->getPortfolioMetricsJson();
            if (reader.parse(jsonStr, metrics)) {
                response["data"] = metrics;
                conn->send(response.toStyledString());
            }
        }
    }
    else if (cmd == "get_transactions") {
        if (engine_) {
            Json::Value response;
            response["type"] = "transactions";
            Json::Value transactionsArray(Json::arrayValue);
            
            const auto &transactions = engine_->getTransactions();
            for (const auto &tx : transactions) {
                transactionsArray.append(transactionToJson(tx));
            }
            
            response["data"] = transactionsArray;
            conn->send(response.toStyledString());
        }
    }
    else if (cmd == "start_updates") {
        int interval = 1000; // Default 1 second
        if (command.isMember("interval_ms")) {
            interval = command["interval_ms"].asInt();
        }
        
        startPeriodicUpdates(interval);
        
        // Send confirmation
        Json::Value response;
        response["type"] = "updates";
        response["status"] = "started";
        response["interval_ms"] = interval;
        conn->send(response.toStyledString());
    }
    else if (cmd == "stop_updates") {
        stopPeriodicUpdates();
        
        // Send confirmation
        Json::Value response;
        response["type"] = "updates";
        response["status"] = "stopped";
        conn->send(response.toStyledString());
    }
    else {
        // Unknown command
        Json::Value response;
        response["type"] = "error";
        response["message"] = "Unknown command: " + cmd;
        conn->send(response.toStyledString());
    }
}

void MyWebSocketController::subscribeClient(const WebSocketConnectionPtr &conn, const std::string &stream)
{
    std::lock_guard<std::mutex> lock(subMutex_);
    subscriptions_[stream].insert(conn);
    
    if (debug_) spdlog::info("[WebSocket] Client subscribed to stream: {}", stream);
}

void MyWebSocketController::unsubscribeClient(const WebSocketConnectionPtr &conn, const std::string &stream)
{
    std::lock_guard<std::mutex> lock(subMutex_);
    
    auto it = subscriptions_.find(stream);
    if (it != subscriptions_.end()) {
        it->second.erase(conn);
        
        if (debug_) spdlog::info("[WebSocket] Client unsubscribed from stream: {}", stream);
    }
}

Json::Value MyWebSocketController::transactionToJson(const BacktestEngine::Transaction &tx)
{
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
