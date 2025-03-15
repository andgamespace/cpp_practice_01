#ifndef MY_WEBSOCKET_CONTROLLER_H
#define MY_WEBSOCKET_CONTROLLER_H

#include <drogon/WebSocketController.h>
#include <drogon/HttpController.h>
#include <json/json.h>
#include <string>
#include <unordered_set>
#include <mutex>

/**
 * @brief MyWebSocketController handles WebSocket connections for real-time data push.
 */
class MyWebSocketController : public drogon::WebSocketController<MyWebSocketController>
{
public:
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

private:
    std::unordered_set<drogon::WebSocketConnectionPtr> connections_;
    std::mutex connMutex_;
};

#endif // MY_WEBSOCKET_CONTROLLER_H
