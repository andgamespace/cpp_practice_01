#include "BacktestEngine.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <taskflow/taskflow.hpp>  // Taskflow uses namespace tf
#include <cmath>

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

        // Process each ticker in its own task (batching per ticker).
        for (auto &pair : tickerData_) {
            const std::string &ticker = pair.first;
            std::shared_ptr<arrow::Table> table = pair.second;
            size_t &index = tickerIndices_[ticker];

            if (index < table->num_rows()) {
                dataRemaining = true;
                taskflow.emplace([this, ticker, table, &index]() {
                    if (strategies_.find(ticker) != strategies_.end()) {
                        auto &strategy = strategies_[ticker];
                        int currentHolding = holdings_[ticker];
                        auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                        if (txOpt.has_value()) {
                            std::lock_guard<std::mutex> lock(mtx_);
                            const auto &tx = txOpt.value();
                            if (tx.action == Action::Buy) {
                                double cost = tx.price * tx.quantity;
                                if (cash_balance_ < cost) {
                                    spdlog::warn("Rejected buy for {}: insufficient funds (cost=${}, cash=${})",
                                                 ticker, cost, cash_balance_);
                                    index++;
                                    return;
                                }
                            }
                            transactions_.push_back(tx);
                            logTransaction(tx);
                            if (tx.action == Action::Buy) {
                                holdings_[ticker] += tx.quantity;
                                cash_balance_ -= tx.price * tx.quantity;
                                buy_prices_[ticker].push_back(tx.price);
                            } else if (tx.action == Action::Sell) {
                                holdings_[ticker] -= tx.quantity;
                                cash_balance_ += tx.price * tx.quantity;
                                if (!buy_prices_[ticker].empty()) {
                                    double buyPrice = buy_prices_[ticker].front();
                                    buy_prices_[ticker].pop_front();
                                    if (tx.price > buyPrice) {
                                        wins_++;
                                    } else {
                                        losses_++;
                                    }
                                }
                            }
                        }
                    }
                    index++;
                });
            }
        }
        executor.run(taskflow).wait();
        iteration++;

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

// === New RL methods ===

// Default step() using registered strategies.
BacktestEngine::StepResult BacktestEngine::step() {
    StepResult result;
    bool anyDataLeft = false;

    for (auto &pair : tickerData_) {
        const std::string &ticker = pair.first;
        std::shared_ptr<arrow::Table> table = pair.second;
        size_t &index = tickerIndices_[ticker];

        if (index < table->num_rows()) {
            anyDataLeft = true;
            if (strategies_.find(ticker) != strategies_.end()) {
                auto &strategy = strategies_[ticker];
                int currentHolding = holdings_[ticker];
                auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                if (txOpt.has_value()) {
                    std::lock_guard<std::mutex> lock(mtx_);
                    const auto &tx = txOpt.value();
                    if (tx.action == Action::Buy) {
                        double cost = tx.price * tx.quantity;
                        if (cash_balance_ >= cost) {
                            transactions_.push_back(tx);
                            logTransaction(tx);
                            holdings_[ticker] += tx.quantity;
                            cash_balance_ -= cost;
                            buy_prices_[ticker].push_back(tx.price);
                        }
                    } else if (tx.action == Action::Sell) {
                        transactions_.push_back(tx);
                        logTransaction(tx);
                        holdings_[ticker] -= tx.quantity;
                        cash_balance_ += tx.price * tx.quantity;
                        if (!buy_prices_[ticker].empty()) {
                            double buyPrice = buy_prices_[ticker].front();
                            buy_prices_[ticker].pop_front();
                            if (tx.price > buyPrice) {
                                wins_++;
                            } else {
                                losses_++;
                            }
                        }
                    }
                }
            }
            auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
            double price = closeArray->Value(index);
            result.observations[ticker] = price;
            index++;
        } else {
            result.observations[ticker] = -1.0;
        }
    }
    result.done = !anyDataLeft;
    result.reward = (cash_balance_ - initial_balance_) / initial_balance_;
    // TODO: Implement risk-adjusted reward metrics (e.g., Sharpe ratio) in the future.
    return result;
}

// External action version of step()
// externalActions: key = ticker, value = number of shares to trade (positive: buy, negative: sell)
BacktestEngine::StepResult BacktestEngine::step(const std::map<std::string, double> &externalActions) {
    StepResult result;
    bool anyDataLeft = false;

    for (auto &pair : tickerData_) {
        const std::string &ticker = pair.first;
        std::shared_ptr<arrow::Table> table = pair.second;
        size_t &index = tickerIndices_[ticker];

        if (index < table->num_rows()) {
            anyDataLeft = true;
            auto it = externalActions.find(ticker);
            if (it != externalActions.end()) {
                double actionValue = it->second;
                // Buy if actionValue > 0
                if (actionValue > 0) {
                    int buyQty = static_cast<int>(actionValue);
                    if (holdings_[ticker] == 0) { // Only buy if no current holding.
                        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
                        double price = closeArray->Value(index);
                        double cost = price * buyQty;
                        if (cash_balance_ >= cost) {
                            Transaction tx;
                            tx.action = Action::Buy;
                            tx.ticker = ticker;
                            tx.quantity = buyQty;
                            tx.price = price;
                            auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
                            tx.datetime = datetimeArray->GetString(index);
                            {
                                std::lock_guard<std::mutex> lock(mtx_);
                                transactions_.push_back(tx);
                                logTransaction(tx);
                                holdings_[ticker] += buyQty;
                                cash_balance_ -= cost;
                                buy_prices_[ticker].push_back(price);
                            }
                        }
                    }
                }
                // Sell if actionValue < 0
                else if (actionValue < 0) {
                    int sellQty = static_cast<int>(-actionValue);
                    if (holdings_[ticker] > 0) {
                        auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
                        double price = closeArray->Value(index);
                        Transaction tx;
                        tx.action = Action::Sell;
                        tx.ticker = ticker;
                        tx.quantity = std::min(holdings_[ticker], sellQty);
                        tx.price = price;
                        auto datetimeArray = std::static_pointer_cast<arrow::StringArray>(table->column(0)->chunk(0));
                        tx.datetime = datetimeArray->GetString(index);
                        {
                            std::lock_guard<std::mutex> lock(mtx_);
                            transactions_.push_back(tx);
                            logTransaction(tx);
                            holdings_[ticker] -= tx.quantity;
                            cash_balance_ += price * tx.quantity;
                            if (!buy_prices_[ticker].empty()) {
                                buy_prices_[ticker].clear();
                            }
                        }
                    }
                }
                // If actionValue == 0, do nothing.
            }
            else {
                // No external action; use default strategy.
                if (strategies_.find(ticker) != strategies_.end()) {
                    auto &strategy = strategies_[ticker];
                    int currentHolding = holdings_[ticker];
                    auto txOpt = strategy->onTick(ticker, table, index, currentHolding);
                    if (txOpt.has_value()) {
                        std::lock_guard<std::mutex> lock(mtx_);
                        const auto &tx = txOpt.value();
                        if (tx.action == Action::Buy) {
                            double cost = tx.price * tx.quantity;
                            if (cash_balance_ >= cost) {
                                transactions_.push_back(tx);
                                logTransaction(tx);
                                holdings_[ticker] += tx.quantity;
                                cash_balance_ -= cost;
                                buy_prices_[ticker].push_back(tx.price);
                            }
                        } else if (tx.action == Action::Sell) {
                            transactions_.push_back(tx);
                            logTransaction(tx);
                            holdings_[ticker] -= tx.quantity;
                            cash_balance_ += tx.price * tx.quantity;
                            if (!buy_prices_[ticker].empty()) {
                                double buyPrice = buy_prices_[ticker].front();
                                buy_prices_[ticker].pop_front();
                                if (tx.price > buyPrice) {
                                    wins_++;
                                } else {
                                    losses_++;
                                }
                            }
                        }
                    }
                }
            }
            auto closeArray = std::static_pointer_cast<arrow::DoubleArray>(table->column(4)->chunk(0));
            double price = closeArray->Value(index);
            result.observations[ticker] = price;
            index++;
        }
        else {
            result.observations[ticker] = -1.0;
        }
    }
    result.done = !anyDataLeft;
    result.reward = (cash_balance_ - initial_balance_) / initial_balance_;
    return result;
}

void BacktestEngine::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &pair : tickerIndices_) {
        pair.second = 0;
    }
    holdings_.clear();
    transactions_.clear();
    for (auto &pair : buy_prices_) {
        pair.second.clear();
    }
    wins_ = 0;
    losses_ = 0;
    cash_balance_ = initial_balance_;
    for (const auto &pair : tickerData_) {
        holdings_[pair.first] = 0;
    }
}
