#include "tensorflow/compiler/xla/tos_cpu_kernel.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

#include <iostream>

namespace tos {
namespace {

namespace py = pybind11;

std::vector<Data> _to_data(
  std::vector<py::buffer> items)
{
  std::vector<Data> ret;
  ret.reserve(items.size());
  for(py::buffer& b: items) {
    py::buffer_info info = b.request();
    size_t sz = info.itemsize;
    for(auto const& s: info.shape) {
      sz *= s;
    }
    ret.push_back(Data{.data = (float*)info.ptr, .size = sz});
  }
  for(auto const& d: ret) {
    std::cout << d.data << ", " << d.size << std::endl;
  }
  return ret;
}

struct Executor {
  Executor():
    scratch_buffer(nullptr),
    scratch_buffer_size(0)
  {
    kernels.reserve(500);
  }

  size_t register_kernel(std::string const& hlo) {
    kernels.emplace_back(hlo);

    size_t sz = kernels.back().scratch_buffer_size();
    if(sz > scratch_buffer_size) {
      scratch_buffer_size = sz;
    }

    return kernels.size()-1;
  }

  void start() { allocate_scratch_buffer(); }

  void stop()  { deallocate_scratch_buffer(); }

  void call(
    size_t which_kernel,
    std::vector<py::buffer> const& inns,
    std::vector<py::buffer> const& outs) const
  {
    CpuKernel const& kernel = kernels[which_kernel];
    kernel(_to_data(inns), _to_data(outs), scratch_buffer);
  }

private:
  std::vector<CpuKernel> kernels;
  void*  scratch_buffer;
  size_t scratch_buffer_size;

  void allocate_scratch_buffer() {
    if(scratch_buffer_size > 0) {
      scratch_buffer = new char[scratch_buffer_size];
    }
  }

  void deallocate_scratch_buffer() {
    if(scratch_buffer != nullptr) {
      char* c = (char*)scratch_buffer;
      delete[] c;
    }
  }
};

PYBIND11_MODULE(tos_executor, m) {
  py::class_<Executor>(m, "Executor")
      .def(py::init<>())
      .def("start", &Executor::start)
      .def("stop",  &Executor::stop)
      .def("register_kernel", &Executor::register_kernel)
      .def("call", &Executor::call);
}

}  // namespace
}  // namespace tos
