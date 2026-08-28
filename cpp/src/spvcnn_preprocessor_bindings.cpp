#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "spvcnn_preprocessor.hpp"

namespace py = pybind11;
using namespace foveated_mapping;

void init_spvcnn_preprocessor_bindings(py::module_& m) {
    py::class_<SPVCNNPreprocessor>(m, "SPVCNNPreprocessor")
        .def(py::init<float>(), py::arg("voxel_size") = 0.05f)
        .def("quantize_and_index", [](
            const SPVCNNPreprocessor& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> points
        ) {
            auto r_pts = points.unchecked<2>();
            size_t n = r_pts.shape(0);
            size_t stride = r_pts.shape(1);

            auto res = self.quantize_and_index(r_pts.data(0, 0), n, stride);
            size_t m = res.num_voxels;

            py::array_t<int64_t> out_coords({m, static_cast<size_t>(3)});
            std::memcpy(out_coords.mutable_data(), res.voxel_coords.data(), sizeof(int64_t) * m * 3);

            py::array_t<int64_t> out_p2v({n});
            std::memcpy(out_p2v.mutable_data(), res.point_to_voxel_idx.data(), sizeof(int64_t) * n);

            py::array_t<int64_t> out_v2p({m});
            std::memcpy(out_v2p.mutable_data(), res.voxel_to_point_idx.data(), sizeof(int64_t) * m);

            return py::make_tuple(
                out_coords,
                out_p2v,
                out_v2p,
                m
            );
        });
}
