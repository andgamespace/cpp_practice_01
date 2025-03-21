#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <drogon/drogon.h>
#include <thread>
#include <future>

// Optional PyTorch support
#if PYTORCH_AVAILABLE
#include <torch/torch.h>
#endif

#include "BacktestEngine.h"
#include "DataLoader.h"
#include "MyWebSocketController.h"
#include "FrontendController.h"
#include "TechnicalIndicators.h"

namespace py = pybind11;

// Global variables for server management
std::thread server_thread;
bool server_running = false;
std::shared_ptr<MyWebSocketController> ws_controller;

// Trampoline class for Strategy to allow Python subclassing.
class PyStrategy : public BacktestEngine::Strategy {
public:
    using BacktestEngine::Strategy::Strategy;

    std::optional<BacktestEngine::Transaction> onTick(const std::string &ticker,
                                                     const std::shared_ptr<arrow::Table> &table,
                                                     size_t currentIndex,
                                                     int currentHolding) override {
        PYBIND11_OVERRIDE_PURE(
            std::optional<BacktestEngine::Transaction>,
            BacktestEngine::Strategy,
            onTick,
            ticker, table, currentIndex, currentHolding
        );
    }
};

// Function to start the web server in a separate thread
void start_server(int port, BacktestEngine* engine) {
    if (server_running) {
        spdlog::warn("Server is already running");
        return;
    }
    
    // Create WebSocket controller
    ws_controller = std::make_shared<MyWebSocketController>();
    
    // Set up the engine to broadcast to WebSocket
    if (engine) {
        engine->setJsonBroadcastCallback([](const std::string& json) {
            if (ws_controller) {
                Json::Value root;
                Json::Reader reader;
                if (reader.parse(json, root)) {
                    root["type"] = "portfolio_update";
                    ws_controller->broadcastJson(root);
                }
            }
        });
    }
    
    // Configure and start Drogon
    drogon::app().addListener("0.0.0.0", port);
    
    // Start server in a separate thread
    server_thread = std::thread([]() {
        server_running = true;
        drogon::app().run();
    });
    
    spdlog::info("Web server started on port {}", port);
}

// Function to stop the web server
void stop_server() {
    if (!server_running) {
        spdlog::warn("Server is not running");
        return;
    }
    
    drogon::app().quit();
    if (server_thread.joinable()) {
        server_thread.join();
    }
    server_running = false;
    ws_controller.reset();
    spdlog::info("Web server stopped");
}

#if PYTORCH_AVAILABLE
// Convert numpy array to torch tensor
torch::Tensor numpy_to_torch(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    auto tensor = torch::from_blob(buf.ptr, {buf.shape.begin(), buf.shape.end()}, torch::kFloat32);
    return tensor.clone(); // Clone to own the memory
}

// Convert torch tensor to numpy array
py::array_t<float> torch_to_numpy(const torch::Tensor& tensor) {
    tensor = tensor.contiguous();
    return py::array_t<float>(
        tensor.sizes().vec(),
        {tensor.strides().begin(), tensor.strides().end()},
        tensor.data_ptr<float>()
    );
}
#endif

PYBIND11_MODULE(my_module, m) {
    m.doc() = "Python bindings for the trading environment with RL interface";

    // Enum for transaction actions
    py::enum_<BacktestEngine::Action>(m, "Action")
        .value("Buy", BacktestEngine::Action::Buy)
        .value("Sell", BacktestEngine::Action::Sell)
        .value("Hold", BacktestEngine::Action::Hold)
        .export_values();

    // Transaction struct
    py::class_<BacktestEngine::Transaction>(m, "Transaction")
        .def(py::init<>())
        .def_readwrite("action", &BacktestEngine::Transaction::action)
        .def_readwrite("ticker", &BacktestEngine::Transaction::ticker)
        .def_readwrite("quantity", &BacktestEngine::Transaction::quantity)
        .def_readwrite("price", &BacktestEngine::Transaction::price)
        .def_readwrite("datetime", &BacktestEngine::Transaction::datetime);

    // Performance metrics struct
    py::class_<BacktestEngine::PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("initialBalance", &BacktestEngine::PerformanceMetrics::initialBalance)
        .def_readwrite("finalBalance", &BacktestEngine::PerformanceMetrics::finalBalance)
        .def_readwrite("totalReturn", &BacktestEngine::PerformanceMetrics::totalReturn)
        .def_readwrite("annualizedReturn", &BacktestEngine::PerformanceMetrics::annualizedReturn)
        .def_readwrite("sharpeRatio", &BacktestEngine::PerformanceMetrics::sharpeRatio)
        .def_readwrite("maxDrawdown", &BacktestEngine::PerformanceMetrics::maxDrawdown)
        .def_readwrite("totalTrades", &BacktestEngine::PerformanceMetrics::totalTrades)
        .def_readwrite("winningTrades", &BacktestEngine::PerformanceMetrics::winningTrades)
        .def_readwrite("losingTrades", &BacktestEngine::PerformanceMetrics::losingTrades)
        .def_readwrite("winRate", &BacktestEngine::PerformanceMetrics::winRate)
        .def_readwrite("profitFactor", &BacktestEngine::PerformanceMetrics::profitFactor)
        .def_readwrite("averageWin", &BacktestEngine::PerformanceMetrics::averageWin)
        .def_readwrite("averageLoss", &BacktestEngine::PerformanceMetrics::averageLoss)
        .def_readwrite("expectancy", &BacktestEngine::PerformanceMetrics::expectancy);

    // StepResult struct
    py::class_<BacktestEngine::StepResult>(m, "StepResult")
        .def(py::init<>())
        .def_readwrite("observations", &BacktestEngine::StepResult::observations)
        .def_readwrite("reward", &BacktestEngine::StepResult::reward)
        .def_readwrite("done", &BacktestEngine::StepResult::done)
        .def_readwrite("features", &BacktestEngine::StepResult::features);

    // Strategy base class
    py::class_<BacktestEngine::Strategy, PyStrategy>(m, "Strategy")
        .def(py::init<>())
        .def("onTick", &BacktestEngine::Strategy::onTick);

    // DataLoader class
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<>())
        .def("loadTickerData", &DataLoader::loadTickerData)
        .def("loadMultipleTickers", &DataLoader::loadMultipleTickers)
        .def("getTickerData", &DataLoader::getTickerData)
        .def("getAllTickerData", &DataLoader::getAllTickerData)
        .def("hasTickerData", &DataLoader::hasTickerData)
        .def("getAvailableTickers", &DataLoader::getAvailableTickers)
        .def("setDebugMode", &DataLoader::setDebugMode)
        .def("printTableHead", [](DataLoader& self, const std::string& ticker, int n) {
            auto table = self.getTickerData(ticker);
            if (!table) {
                throw std::runtime_error("Ticker data not found: " + ticker);
            }
            
            // Print schema
            std::cout << "Schema: " << table->schema()->ToString() << std::endl;
            
            // Print column names
            std::cout << "Columns: ";
            for (int i = 0; i < table->num_columns(); ++i) {
                std::cout << table->schema()->field(i)->name();
                if (i < table->num_columns() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            // Print rows
            int rows = std::min(static_cast<int>(table->num_rows()), n);
            for (int i = 0; i < rows; ++i) {
                std::cout << "Row " << i << ": ";
                for (int j = 0; j < table->num_columns(); ++j) {
                    auto column = table->column(j);
                    if (j == 0) { // datetime column (string)
                        auto array = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));
                        std::cout << array->GetString(i);
                    } else { // numeric columns
                        auto array = std::static_pointer_cast<arrow::DoubleArray>(column->chunk(0));
                        std::cout << array->Value(i);
                    }
                    
                    if (j < table->num_columns() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
            
            return rows;
        }, "Print the first n rows of a ticker's data table", py::arg("ticker"), py::arg("n") = 5)
        .def("arrowToPandas", [](DataLoader& self, const std::string& ticker) {
            auto table = self.getTickerData(ticker);
            if (!table) {
                throw std::runtime_error("Ticker data not found: " + ticker);
            }
            
            // Import pandas
            py::object pd = py::module::import("pandas");
            
            // Create a dictionary to hold the data
            py::dict data;
            
            // Process each column
            for (int i = 0; i < table->num_columns(); ++i) {
                std::string colName = table->schema()->field(i)->name();
                auto column = table->column(i);
                
                if (i == 0) { // datetime column (string)
                    auto array = std::static_pointer_cast<arrow::StringArray>(column->chunk(0));
                    py::list values;
                    for (int64_t j = 0; j < array->length(); ++j) {
                        values.append(array->GetString(j));
                    }
                    data[colName.c_str()] = values;
                } else { // numeric columns
                    auto array = std::static_pointer_cast<arrow::DoubleArray>(column->chunk(0));
                    py::list values;
                    for (int64_t j = 0; j < array->length(); ++j) {
                        values.append(array->Value(j));
                    }
                    data[colName.c_str()] = values;
                }
            }
            
            // Create pandas DataFrame
            return pd.attr("DataFrame")(data);
        }, "Convert Arrow table to pandas DataFrame", py::arg("ticker"));

    // BacktestEngine class
    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<>())
        .def("reset", &BacktestEngine::reset)
        .def("step", (BacktestEngine::StepResult (BacktestEngine::*)()) &BacktestEngine::step)
        .def("step_with_action", (BacktestEngine::StepResult (BacktestEngine::*)(const std::map<std::string, double>&)) &BacktestEngine::step)
#if PYTORCH_AVAILABLE
        .def("step_with_tensor", [](BacktestEngine& self, py::array_t<float> array) {
            torch::Tensor tensor = numpy_to_torch(array);
            return self.stepWithTensor(tensor);
        })
#endif
        .def("getPortfolioMetrics", &BacktestEngine::getPortfolioMetrics)
        .def("getPortfolioMetricsJson", &BacktestEngine::getPortfolioMetricsJson)
        .def("getPerformanceMetrics", &BacktestEngine::getPerformanceMetrics)
        .def("getCashBalance", &BacktestEngine::getCashBalance)
        .def("getHoldings", &BacktestEngine::getHoldings)
        .def("getTransactions", &BacktestEngine::getTransactions)
        .def("getObservationDimension", &BacktestEngine::getObservationDimension)
        .def("getActionDimension", &BacktestEngine::getActionDimension)
        .def("getAvailableTickers", &BacktestEngine::getAvailableTickers)
        .def("setDebugMode", &BacktestEngine::setDebugMode)
        .def("addTickerData", &BacktestEngine::addTickerData)
        .def("setTickerData", [](BacktestEngine& self, const std::map<std::string, std::shared_ptr<arrow::Table>>& data) {
            self.setTickerData(data);
        })
        .def("registerStrategy", [](BacktestEngine& self, const std::string& ticker, BacktestEngine::Strategy* strategy) {
            self.registerStrategy(ticker, std::unique_ptr<BacktestEngine::Strategy>(strategy));
        }, py::keep_alive<1, 3>())
        .def("runBacktest", &BacktestEngine::runBacktest)
        .def("setBroadcastCallback", &BacktestEngine::setBroadcastCallback)
        .def("setJsonBroadcastCallback", &BacktestEngine::setJsonBroadcastCallback);

    // Server management functions
    m.def("start_server", &start_server, "Start the web server for visualization",
          py::arg("port") = 8080, py::arg("engine") = nullptr);
    m.def("stop_server", &stop_server, "Stop the web server");
    m.def("is_server_running", []() { return server_running; }, "Check if the server is running");

    // TechnicalIndicators class
    py::class_<TechnicalIndicators>(m, "TechnicalIndicators")
        .def(py::init<>())
        .def("setDebugMode", &TechnicalIndicators::setDebugMode)
        .def("calculateSMA", &TechnicalIndicators::calculateSMA,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("newColumnName") = py::none())
        .def("calculateEMA", &TechnicalIndicators::calculateEMA,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("newColumnName") = py::none())
        .def("calculateRSI", &TechnicalIndicators::calculateRSI,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("newColumnName") = py::none())
        .def("calculateBollingerBands", &TechnicalIndicators::calculateBollingerBands,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("stdDev") = 2.0,
             py::arg("upperBandName") = py::none(), py::arg("middleBandName") = py::none(), py::arg("lowerBandName") = py::none())
        .def("calculateMACD", &TechnicalIndicators::calculateMACD,
             py::arg("table"), py::arg("column"), py::arg("fastPeriod") = 12, py::arg("slowPeriod") = 26, py::arg("signalPeriod") = 9,
             py::arg("macdName") = py::none(), py::arg("signalName") = py::none(), py::arg("histogramName") = py::none())
        .def("calculateATR", &TechnicalIndicators::calculateATR,
             py::arg("table"), py::arg("highColumn"), py::arg("lowColumn"), py::arg("closeColumn"), py::arg("period") = 14,
             py::arg("newColumnName") = py::none())
        .def("calculateROC", &TechnicalIndicators::calculateROC,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("newColumnName") = py::none())
        .def("calculateStdDev", &TechnicalIndicators::calculateStdDev,
             py::arg("table"), py::arg("column"), py::arg("period"), py::arg("newColumnName") = py::none());

    // Utility functions
    m.def("load_stock_data", [](const std::string& base_dir, const std::vector<std::string>& tickers) {
        DataLoader loader;
        std::unordered_map<std::string, std::vector<std::string>> tickerFilePaths;
        
        for (const auto& ticker : tickers) {
            std::vector<std::string> filePaths = {
                base_dir + "/time-series-" + ticker + "-5min.csv",
                base_dir + "/time-series-" + ticker + "-5min(1).csv",
                base_dir + "/time-series-" + ticker + "-5min(2).csv"
            };
            tickerFilePaths[ticker] = filePaths;
        }
        
        int loaded = loader.loadMultipleTickers(tickerFilePaths);
        if (loaded == 0) {
            throw std::runtime_error("Failed to load any ticker data");
        }
        
        return loader.getAllTickerData();
    }, "Load stock data for multiple tickers", py::arg("base_dir"), py::arg("tickers"));
    
    // Helper function to add technical indicators to a ticker's data
    m.def("add_technical_indicators", [](DataLoader& loader, const std::string& ticker,
                                        const std::vector<std::string>& indicators,
                                        const std::map<std::string, std::map<std::string, std::string>>& params) {
        auto table = loader.getTickerData(ticker);
        if (!table) {
            throw std::runtime_error("Ticker data not found: " + ticker);
        }
        
        TechnicalIndicators ti;
        
        // Convert params to the format expected by calculateMultipleIndicators
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> paramsMap;
        for (const auto& [key, value] : params) {
            std::unordered_map<std::string, std::string> innerMap;
            for (const auto& [innerKey, innerValue] : value) {
                innerMap[innerKey] = innerValue;
            }
            paramsMap[key] = innerMap;
        }
        
        auto resultTable = ti.calculateMultipleIndicators(table, indicators, paramsMap);
        
        // Update the ticker data in the loader
        loader.updateTickerData(ticker, resultTable);
        
        return resultTable;
    }, "Add technical indicators to a ticker's data",
       py::arg("loader"), py::arg("ticker"), py::arg("indicators"), py::arg("params"));
}
