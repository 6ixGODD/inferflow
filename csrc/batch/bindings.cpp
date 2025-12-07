#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inferflow/batch/batcher.hpp"

namespace py = pybind11;

namespace inferflow {
namespace batch {

void BindBatchModule(py::module_& parent) {
  auto batch =
      parent.def_submodule("batch", "High-performance batch processing");

  // Config struct
  py::class_<DynamicBatcher::Config>(batch, "BatcherConfig")
      .def(py::init<>())
      .def_readwrite("min_batch_size", &DynamicBatcher::Config::min_batch_size)
      .def_readwrite("max_batch_size", &DynamicBatcher::Config::max_batch_size)
      .def_readwrite("max_wait_ms", &DynamicBatcher::Config::max_wait_ms)
      .def_readwrite("queue_capacity", &DynamicBatcher::Config::queue_capacity)
      .def_readwrite("block_on_full", &DynamicBatcher::Config::block_on_full);

  // Metrics snapshot (copyable)
  py::class_<BatchMetricsSnapshot>(batch, "BatchMetrics")
      .def_readonly("total_requests", &BatchMetricsSnapshot::total_requests)
      .def_readonly("total_batches", &BatchMetricsSnapshot::total_batches)
      .def_readonly("rejected_requests",
                    &BatchMetricsSnapshot::rejected_requests)
      .def_readonly("current_queue_size",
                    &BatchMetricsSnapshot::current_queue_size)
      .def_readonly("current_batch_size",
                    &BatchMetricsSnapshot::current_batch_size)
      .def_readonly("total_latency_ms", &BatchMetricsSnapshot::total_latency_ms)
      .def_readonly("avg_batch_size", &BatchMetricsSnapshot::avg_batch_size)
      .def_readonly("avg_latency_ms", &BatchMetricsSnapshot::avg_latency_ms);

  // DynamicBatcher class
  py::class_<DynamicBatcher>(batch, "DynamicBatcher")
      .def(py::init<const DynamicBatcher::Config&>(), py::arg("config"))
      .def("start", &DynamicBatcher::Start, py::arg("runtime"),
           py::call_guard<py::gil_scoped_release>(),
           "Start the batcher with a runtime")
      .def("stop", &DynamicBatcher::Stop,
           py::call_guard<py::gil_scoped_release>(), "Stop the batcher")
      .def("submit", &DynamicBatcher::Submit, py::arg("item"),
           "Submit an item for batched processing")
      .def("is_running", &DynamicBatcher::IsRunning,
           "Check if batcher is running")
      .def("get_metrics", &DynamicBatcher::GetMetrics, "Get current metrics")
      .def("queue_size", &DynamicBatcher::QueueSize, "Get current queue size");
}

}  // namespace batch
}  // namespace inferflow
