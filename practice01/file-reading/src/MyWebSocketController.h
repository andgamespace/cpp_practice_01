#ifndef MY_WEBSOCKET_CONTROLLER_H
#define MY_WEBSOCKET_CONTROLLER_H

#include <drogon/WebSocketController.h>
#include <drogon/HttpController.h>
#include <json/json.h>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "BacktestEngine.h"

/**
 * @brief MyWebSocketController handles WebSocket connections for real-time data push.
 * Provides real-time updates from the BacktestEngine to connected clients.
 */
class MyWebSocketController : public drogon::WebSocketController<MyWebSocketController>
{
public:
    MyWebSocketController();
    
    // Set the BacktestEngine to use for real-time updates
    void setBacktestEngine(std::shared_ptr<BacktestEngine> engine);

    // Called when a new WebSocket connection is established.
    void handleNewConnection(const drogon::HttpRequestPtr &req,
                             const drogon::WebSocketConnectionPtr &conn) override;

    // Called when a WebSocket message arrives.
    void handleNewMessage(const drogon::WebSocketConnectionPtr &conn,
                          std::string &&message,
                          const drogon::WebSocketMessageType &type) override;

    // Called when a WebSocket connection is closed.
    void handleConnectionClosed(const drogon::WebSocketConnectionPtr &conn) override;

    // Define the WebSocket route.
    WS_PATH_LIST_BEGIN
        WS_PATH_ADD("/ws", "GET");
    WS_PATH_LIST_END

    // Broadcast JSON data to all connected clients.
    void broadcastJson(const Json::Value &data);
    
    // Broadcast a string message to all connected clients.
    void broadcastMessage(const std::string &message);
    
    // Broadcast portfolio metrics to all connected clients.
    void broadcastPortfolioMetrics();
    
    // Start periodic updates
    void startPeriodicUpdates(int intervalMs = 1000);
    
    // Stop periodic updates
    void stopPeriodicUpdates();

private:
    std::unordered_set<drogon::WebSocketConnectionPtr> connections_;
    std::mutex connMutex_;
    std::shared_ptr<BacktestEngine> engine_;
    bool debug_;
    
    // Client subscription management
    std::unordered_map<std::string, std::unordered_set<drogon::WebSocketConnectionPtr>> subscriptions_;
    std::mutex subMutex_;
    
    // Periodic update management
    bool periodicUpdatesRunning_;
    std::mutex periodicMutex_;
    std::thread periodicThread_;
    
    // Process client commands
    void processCommand(const drogon::WebSocketConnectionPtr &conn, const Json::Value &command);
    
    // Subscribe a client to a specific data stream
    void subscribeClient(const drogon::WebSocketConnectionPtr &conn, const std::string &stream);
    
    // Unsubscribe a client from a specific data stream
    void unsubscribeClient(const drogon::WebSocketConnectionPtr &conn, const std::string &stream);
    
    // Convert transaction to JSON
    Json::Value transactionToJson(const BacktestEngine::Transaction &tx);
};

#endif // MY_WEBSOCKET_CONTROLLER_H
