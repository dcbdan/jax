#include "tensorflow/compiler/xla/tos_cpu_kernel.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"

namespace tos {
namespace {

namespace py = pybind11;

std::vector<Data> _to_data(
  std::vector<py::buffer>& items)
{
  std::vector<Data> ret;
  ret.reserve(items.size());
  for(py::buffer const& b: items) {
    py::buffer_info info = b.request();
    assert(info.itemsize == sizeof(float));
    size_t sz = 1;
    for(auto const& s: info.shape) {
      sz *= s;
    }
    ret.push_back(Data{.data = (float*)info.ptr, .size = sz});
  }

  return ret;
}

struct Executor {
  Executor():
    holder(new RunOptionsHolder()),
    scratch_buffer(nullptr),
    allocated(0),
    scratch_buffer_size(0)
  {
    kernels.reserve(500);
  }

  ~Executor() { deallocate(); }

  size_t register_kernel(std::string const& hlo) {
    kernels.emplace_back(new CpuKernel(hlo));

    size_t sz = kernels.back()->scratch_buffer_size();
    if(sz > scratch_buffer_size) {
      scratch_buffer_size = sz;
    }

    return kernels.size()-1;
  }

  void allocate() {
    if(allocated < scratch_buffer_size) {
      deallocate();
      scratch_buffer = new char[scratch_buffer_size];
      allocated = scratch_buffer_size;
    }
  }

  void deallocate() {
    if(scratch_buffer != nullptr) {
      char* c = (char*)scratch_buffer;
      delete[] c;
    }
    scratch_buffer = nullptr;
    allocated = 0;
  }

  void call(
    size_t which_kernel,
    std::vector<py::buffer> inns,
    std::vector<py::buffer> outs)
  {
    CpuKernel& kernel = *(kernels[which_kernel]);
    if(allocated >= kernel.scratch_buffer_size()) {
      kernel(holder->get_run_options(), _to_data(inns), _to_data(outs), scratch_buffer);
    } else {
      kernel(holder->get_run_options(), _to_data(inns), _to_data(outs), nullptr);
    }
  }

private:
  std::unique_ptr<RunOptionsHolder> holder;
  std::vector<std::unique_ptr<CpuKernel>> kernels;
  void*  scratch_buffer;
  size_t allocated;
  size_t scratch_buffer_size;
};

size_t _mem_align_offset(void* mem_, size_t align_to) {
  static_assert(sizeof(size_t) >= sizeof(std::uintptr_t));
  size_t mem = reinterpret_cast<std::uintptr_t>(mem_);

  // An easier to understand but slower way to do it...
  //    for(int i = 0; i != align_to; ++i) {
  //      if((mem + i) % align_to == 0) {
  //        return i;
  //      }
  //    }

  return ((mem + (align_to-1)) & ~(align_to-1)) - mem;
}
size_t mem_align_offset(py::buffer b, size_t align_to) {
  py::buffer_info info = b.request();
  return _mem_align_offset(info.ptr, align_to); 
}

PYBIND11_MODULE(tos_executor, m) {
  py::class_<Executor>(m, "Executor")
      .def(py::init<>())
      .def("allocate", &Executor::allocate)
      .def("deallocate", &Executor::deallocate)
      .def("register_kernel", &Executor::register_kernel)
      .def("call", &Executor::call);
  m.def("mem_align_offset", &mem_align_offset);
}

}  // namespace
}  // namespace tos
