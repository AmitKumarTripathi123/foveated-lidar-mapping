#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "grid_rasterizer.hpp"

namespace py = pybind11;
using namespace foveated_mapping;

PYBIND11_MODULE(foveated_grid_cpp, m) {
    m.doc() = "Native C++ 2.5D GridMap Rasterization Accelerator (SIH PS 26130)";

    py::class_<GridConfig>(m, "GridConfig")
        .def(py::init<float, float, float, float, float>(),
             py::arg("min_x") = -50.0f,
             py::arg("max_x") = 50.0f,
             py::arg("min_y") = -50.0f,
             py::arg("max_y") = 50.0f,
             py::arg("resolution") = 0.20f)
        .def_readwrite("min_x", &GridConfig::min_x)
        .def_readwrite("max_x", &GridConfig::max_x)
        .def_readwrite("min_y", &GridConfig::min_y)
        .def_readwrite("max_y", &GridConfig::max_y)
        .def_readwrite("resolution", &GridConfig::resolution)
        .def_readwrite("width", &GridConfig::width)
        .def_readwrite("height", &GridConfig::height);

    py::class_<NativeGridRasterizer>(m, "NativeGridRasterizer")
        .def(py::init<const GridConfig&>(), py::arg("config") = GridConfig())
        .def("rasterize", [](
            const NativeGridRasterizer& self,
            py::array_t<float, py::array::c_style | py::array::forcecast> xyz,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> classes,
            py::array_t<float, py::array::c_style | py::array::forcecast> confidences
        ) {
            auto r_xyz = xyz.unchecked<2>();
            auto r_cls = classes.unchecked<1>();
            auto r_conf = confidences.unchecked<1>();

            size_t n = r_xyz.shape(0);
            auto res = self.rasterize(r_xyz.data(0, 0), r_cls.data(0), r_conf.data(0), n);

            int h = res.height;
            int w = res.width;

            py::array_t<float> py_elev_mean({h, w});
            py::array_t<float> py_elev_min({h, w});
            py::array_t<float> py_elev_max({h, w});
            py::array_t<int64_t> py_sem({h, w});
            py::array_t<float> py_conf({h, w});
            py::array_t<float> py_trav({h, w});
            py::array_t<int32_t> py_cnt({h, w});

            std::memcpy(py_elev_mean.mutable_data(), res.elevation_mean.data(), sizeof(float) * h * w);
            std::memcpy(py_elev_min.mutable_data(), res.elevation_min.data(), sizeof(float) * h * w);
            std::memcpy(py_elev_max.mutable_data(), res.elevation_max.data(), sizeof(float) * h * w);
            std::memcpy(py_sem.mutable_data(), res.semantic_layer.data(), sizeof(int64_t) * h * w);
            std::memcpy(py_conf.mutable_data(), res.confidence_layer.data(), sizeof(float) * h * w);
            std::memcpy(py_trav.mutable_data(), res.traversability_layer.data(), sizeof(float) * h * w);
            std::memcpy(py_cnt.mutable_data(), res.point_count_layer.data(), sizeof(int32_t) * h * w);

            return py::make_tuple(
                py_elev_min,
                py_elev_max,
                py_elev_mean,
                py_sem,
                py_conf,
                py_trav,
                py_cnt
            );
        });
}
