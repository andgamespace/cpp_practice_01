#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "BacktestEngine.h"

namespace py = pybind11;

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

PYBIND11_MODULE(my_module, m) {
    m.doc() = "Python bindings for the BacktestEngine with RL interface";

    py::class_<BacktestEngine::StepResult>(m, "StepResult")
        .def_readwrite("observations", &BacktestEngine::StepResult::observations)
        .def_readwrite("reward", &BacktestEngine::StepResult::reward)
        .def_readwrite("done", &BacktestEngine::StepResult::done);

    // Expose Strategy base class and allow Python subclasses.
    py::class_<BacktestEngine::Strategy, PyStrategy>(m, "Strategy")
        .def(py::init<>())
        .def("onTick", &BacktestEngine::Strategy::onTick);

    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<>())
        .def("reset", &BacktestEngine::reset)
        .def("step", (BacktestEngine::StepResult (BacktestEngine::*)()) &BacktestEngine::step)
        .def("step_with_action", (BacktestEngine::StepResult (BacktestEngine::*)(const std::map<std::string, double>&)) &BacktestEngine::step)
        .def("getPortfolioMetrics", &BacktestEngine::getPortfolioMetrics);
}
