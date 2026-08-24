#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "foveation_accelerator.hpp"

namespace py = pybind11;
using namespace foveated_mapping;

void init_foveation_bindings(py::module_& m) {
    py::class_<FoveationConfig>(m, "FoveationConfig")
        .def(py::init<float, float, float, float, float, float, float>(),
             py::arg("min_range") = 0.5f,
             py::arg("near_dist") = 10.0f,
             py::arg("near_voxel") = 0.05f,
             py::arg("mid_dist") = 40.0f,
             py::arg("mid_voxel") = 0.15f,
             py::arg("far_dist") = 100.0f,
             py::arg("far_voxel") = 0.50f)
        .def_readwrite("min_range", &FoveationConfig::min_range)
        .def_readwrite("near_dist", &FoveationConfig::near_dist)
        .def_readwrite("near_voxel", &FoveationConfig::near_voxel)
        .def_readwrite("mid_dist", &FoveationConfig::mid_dist)
        .def_readwrite("mid_voxel", &FoveationConfig::mid_voxel)
        .def_readwrite("far_dist", &FoveationConfig::far_dist)
        .def_readwrite("far_voxel", &FoveationConfig::far_voxel);

    py::class_<ZoneStats>(m, "ZoneStats")
        .def_readonly("zone_name", &ZoneStats::zone_name)
        .def_readonly("min_dist", &ZoneStats::min_dist)
        .def_readonly("max_dist", &ZoneStats::max_dist)
        .def_readonly("voxel_size", &ZoneStats::voxel_size)
        .def_readonly("input_count", &ZoneStats::input_count)
        .def_readonly("output_count", &ZoneStats::output_count)
        .def_readonly("reduction_pct", &ZoneStats::reduction_pct);

    py::class_<FoveationAccelerator>(m, "FoveationAccelerator")
        .def(py::init<const FoveationConfig&>(), py::arg("config") = FoveationConfig())
        .def("foveate", [](
            const FoveationAccelerator& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> points,
            py::object optional_labels
        ) {
            auto r_pts = points.unchecked<2>();
            size_t n = r_pts.shape(0);

            const int64_t* lbl_ptr = nullptr;
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> py_lbls;
            if (!optional_labels.is_none()) {
                py_lbls = optional_labels.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
                lbl_ptr = py_lbls.data(0);
            }

            auto res = self.foveate(r_pts.data(0, 0), lbl_ptr, n);
            size_t m = res.foveated_count;

            py::array_t<float> out_points({m, static_cast<size_t>(4)});
            std::memcpy(out_points.mutable_data(), res.points.data(), sizeof(float) * m * 4);

            py::object out_labels = py::none();
            if (lbl_ptr != nullptr) {
                py::array_t<int64_t> ret_lbls({m});
                std::memcpy(ret_lbls.mutable_data(), res.labels.data(), sizeof(int64_t) * m);
                out_labels = ret_lbls;
            }

            return py::make_tuple(
                out_points,
                out_labels,
                res.original_count,
                res.foveated_count,
                res.overall_reduction_pct,
                res.filtered_out_count,
                res.zone_stats
            );
        });
}
