#include "MyWebSocketController.h"
#include <drogon/drogon.h>
#include <spdlog/spdlog.h>

using namespace drogon;

void MyWebSocketController::handleNewConnection(const HttpRequestPtr &req,
                                                const WebSocketConnectionPtr &conn)
{
    spdlog::info("[WebSocket] New client connected");

    {
        std::lock_guard<std::mutex> lock(connMutex_);
        connections_.insert(conn);
    }

    // (Optional) Immediately send a welcome message
    Json::Value initMsg;
    initMsg["type"]    = "welcome";
    initMsg["payload"] = "You are connected to Drogon WebSocket!";
    conn->send(initMsg.toStyledString());
}

void MyWebSocketController::handleNewMessage(const WebSocketConnectionPtr &conn,
                                             std::string &&message,
                                             const WebSocketMessageType &type)
{
    if (type == WebSocketMessageType::Text)
    {
        spdlog::info("[WebSocket] Received message: {}", message);

        // Example: parse JSON and echo it back
        Json::Reader reader;
        Json::Value incoming;
        if (reader.parse(message, incoming))
        {
            Json::Value response;
            response["type"]    = "echo";
            response["payload"] = incoming;
            conn->send(response.toStyledString());
        }
    }
    // (Handle Binary if needed)
}

void MyWebSocketController::handleConnectionClosed(const WebSocketConnectionPtr &conn)
{
    spdlog::info("[WebSocket] Client disconnected");
    {
        std::lock_guard<std::mutex> lock(connMutex_);
        connections_.erase(conn);
    }
}

void MyWebSocketController::broadcastJson(const Json::Value &data)
{
    // Convert once
    const std::string out = data.toStyledString();
    std::lock_guard<std::mutex> lock(connMutex_);
    for (auto &c : connections_)
    {
        if (c && c->connected())
        {
            c->send(out);
        }
    }
}
