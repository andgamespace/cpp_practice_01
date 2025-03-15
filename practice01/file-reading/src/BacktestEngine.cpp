#include "BacktestEngine.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <taskflow/taskflow.hpp>  // Taskflow uses namespace tf

BacktestEngine::BacktestEngine()
    : tickerData_{}, tickerIndices_{}, strategies_{}, holdings_{}, transactions_{},
      initial_balance_(100000.0), cash_balance_(100000.0), wins_(0), losses_(0), buy_prices_{},
      broadcast_callback_(nullptr)
{
    spdlog::info("BacktestEngine initialized with starting balance ${}", initial_balance_);
}

BacktestEngine &BacktestEngine::operator=(double new_balance) {
    std::lock_guard<std::mutex> lock(mtx_);
    initial_balance_ = new_balance;
    cash_balance_ = new_balance;
    return *this;
}

void BacktestEngine::addTickerData(const std::string &ticker, const std::shared_ptr<arrow::Table> &table) {
    tickerData_[ticker] = table;
    tickerIndices_[ticker] = 0;
    holdings_[ticker] = 0;
    spdlog::info("Added ticker data for {}", ticker);
}

void BacktestEngine::setTickerData(const std::map<std::string, std::shared_ptr<arrow::Table>> &data) {
    for (const auto &pair : data) {
        addTickerData(pair.first, pair.second);
    }
}

void BacktestEngine::registerStrategy(const std::string &ticker, std::unique_ptr<Strategy> strategy) {
    strategies_[ticker] = std::move(strategy);
    spdlog::info("Registered strategy for ticker {}", ticker);
}

void BacktestEngine::setBroadcastCallback(std::function<void(const std::string &)> callback) {
    std::lock_guard<std::mutex> lock(mtx_);
    broadcast_callback_ = callback;
}

void BacktestEngine::logTransaction(const Transaction &tx) {
    spdlog::info("Transaction: {} {} shares of {} at ${} on {}",
                 (tx.action == Action::Buy ? "BUY" : "SELL"),
                 tx.quantity, tx.ticker, tx.price, tx.datetime);
}

void BacktestEngine::runBacktest() {
    spdlog::info("Starting concurrent backtest simulation");
    tf::Executor executor;
    size_t iteration = 0;
    constexpr size_t updateFrequency = 16; // Broadcast/log every 16 ticks

    bool dataRemaining = true;
    while (dataRemaining) {
        dataRemaining = false;
        tf::Taskflow taskflow;

        for (auto &pair : tickerData_) {
            const std::string &ticker = pair.first;
            std::shared_ptr<arrow::Table> table = pair.second;
            size_t &index = tickerIndices_[ticker];

            // If we haven't exhausted this ticker's data, schedule a task.
            if (index < table->num_rows()) {
                dataRemaining = true;

                taskflow.emplace([this, ticker, table, &index]() {
                    // Check if a strategy is registered.
                    if (strategies_.find(ticker) != strategies_.end()) {
                        auto &strategy = strategies_[ticker];

                        // Lockless read of holdings_ is safe, but we might do a short lock
                        // if we want to ensure consistency. For concurrency, let's do a quick read:
                        int currentHolding = holdings_[ticker];

                        // The strategy decides on a transaction (or none).
                        auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                        if (txOpt.has_value()) {
                            // We lock before modifying shared state:
                            std::lock_guard<std::mutex> lock(mtx_);
                            const auto &tx = txOpt.value();

                            // 1) Check for negative balance if it's a BUY.
                            if (tx.action == Action::Buy) {
                                double cost = tx.price * tx.quantity;
                                if (cash_balance_ < cost) {
                                    // Reject if not enough cash
                                    spdlog::warn("Rejected buy for {}: insufficient funds (cost=${}, cash=${})",
                                                 ticker, cost, cash_balance_);
                                    // skip
                                    index++;
                                    return;
                                }
                            }

                            // If we get here, transaction is accepted
                            transactions_.push_back(tx);
                            logTransaction(tx);

                            if (tx.action == Action::Buy) {
                                holdings_[ticker] += tx.quantity;
                                cash_balance_ -= tx.price * tx.quantity;
                                buy_prices_[ticker].push_back(tx.price);
                            } else if (tx.action == Action::Sell) {
                                holdings_[ticker] -= tx.quantity;
                                cash_balance_ += tx.price * tx.quantity;
                                // Evaluate win/loss
                                if (!buy_prices_[ticker].empty()) {
                                    double buyPrice = buy_prices_[ticker].front();
                                    buy_prices_[ticker].erase(buy_prices_[ticker].begin());
                                    if (tx.price > buyPrice) {
                                        wins_++;
                                    } else {
                                        losses_++;
                                    }
                                }
                            }
                        }
                    }
                    // Move to next row (tick)
                    index++;
                });
            }
        }

        // Execute all tasks concurrently for this iteration
        executor.run(taskflow).wait();
        iteration++;

        // Broadcast or log updates every 'updateFrequency' ticks
        if (iteration % updateFrequency == 0) {
            std::string metrics = getPortfolioMetrics();
            spdlog::info("Iteration {}: {}", iteration, metrics);
            if (broadcast_callback_) {
                broadcast_callback_(metrics);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    spdlog::info("Backtest simulation completed after {} iterations", iteration);
}

std::string BacktestEngine::getPortfolioMetrics() const {
    std::ostringstream oss;
    oss << "Cash: $" << cash_balance_ << ", Holdings: ";
    for (const auto &pair : holdings_) {
        oss << pair.first << "=" << pair.second << " ";
    }
    oss << "| Total Transactions: " << transactions_.size();
    oss << ", Wins: " << wins_ << ", Losses: " << losses_;
    double pct_change = ((cash_balance_ - initial_balance_) / initial_balance_) * 100.0;
    oss << ", % Change: " << pct_change << "%";
    return oss.str();
}
