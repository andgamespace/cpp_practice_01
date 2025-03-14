#ifndef MY_WEBSOCKET_CONTROLLER_H
#define MY_WEBSOCKET_CONTROLLER_H

#include <drogon/WebSocketController.h>
#include <drogon/HttpController.h>  // For drogon::Get if needed
#include <json/json.h>
#include <string>
#include <unordered_set>
#include <mutex>

/**
 * @brief Example WebSocket controller for real-time data push.
 *
 * We rename it to MyWebSocketController to avoid conflict with
 * drogon::WebSocketController class name.
 */
class MyWebSocketController : public drogon::WebSocketController<MyWebSocketController>
{
public:
    // Called when a new WebSocket connection is established
    void handleNewConnection(const drogon::HttpRequestPtr &req,
                             const drogon::WebSocketConnectionPtr &conn) override;

    // Called when a WebSocket message arrives from the client
    void handleNewMessage(const drogon::WebSocketConnectionPtr &conn,
                          std::string &&message,
                          const drogon::WebSocketMessageType &type) override;

    // Called when a WebSocket connection is closed
    void handleConnectionClosed(const drogon::WebSocketConnectionPtr &conn) override;

    // The path for your WebSocket route
    WS_PATH_LIST_BEGIN
        WS_PATH_ADD("/ws", "GET");
    WS_PATH_LIST_END

private:
    // Keep track of all current connections so you can broadcast
    std::unordered_set<drogon::WebSocketConnectionPtr> connections_;
    std::mutex connMutex_;

    // Example: helper to broadcast JSON messages to all clients
    void broadcastJson(const Json::Value &data);
};

#endif // MY_WEBSOCKET_CONTROLLER_H
