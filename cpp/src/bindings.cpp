#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "types.hpp"
#include "foveated_grid.hpp"

namespace py = pybind11;
using namespace foveated_mapping;

namespace {

// Vectorized C++ grid construction directly on NumPy array buffers (Zero-Copy input streaming)
py::dict build_grid_numpy_impl(
    const FoveatedGridEngine& engine,
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    std::optional<py::array_t<int64_t, py::array::c_style | py::array::forcecast>> labels,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> confidences
) {
    py::buffer_info pts_info = points.request();
    if (pts_info.ndim != 2 || pts_info.shape[1] < 3) {
        throw std::invalid_argument("Expected points array of shape (N, 3) or (N, 4)");
    }

    size_t N = pts_info.shape[0];
    size_t feat_dim = pts_info.shape[1];
    const float* pts_ptr = static_cast<const float*>(pts_info.ptr);

    const int64_t* lbls_ptr = nullptr;
    py::buffer_info lbls_info;
    if (labels.has_value()) {
        lbls_info = labels->request();
        if (lbls_info.size < static_cast<ssize_t>(N)) {
            throw std::invalid_argument("Labels array length must match points count");
        }
        lbls_ptr = static_cast<const int64_t*>(lbls_info.ptr);
    }

    const float* confs_ptr = nullptr;
    py::buffer_info confs_info;
    if (confidences.has_value()) {
        confs_info = confidences->request();
        if (confs_info.size < static_cast<ssize_t>(N)) {
            throw std::invalid_argument("Confidences array length must match points count");
        }
        confs_ptr = static_cast<const float*>(confs_info.ptr);
    }

    // Execute optimized core Phase-7 grid engine directly from memory buffer
    std::vector<GridCell> cells = engine.build_grid_raw(pts_ptr, N, feat_dim, lbls_ptr, confs_ptr);
    size_t M = cells.size();

    // Allocate output NumPy arrays
    py::array_t<int64_t> ix_arr(M);
    py::array_t<int64_t> iy_arr(M);
    py::array_t<float> res_arr(M);
    py::array_t<int64_t> count_arr(M);
    py::array_t<float> mean_z_arr(M);
    py::array_t<float> min_z_arr(M);
    py::array_t<float> max_z_arr(M);
    py::array_t<int64_t> class_arr(M);
    py::array_t<float> conf_arr(M);
    py::array_t<float> trav_arr(M);
    std::vector<std::string> bands_list;
    bands_list.reserve(M);

    auto ix_mut = ix_arr.mutable_unchecked<1>();
    auto iy_mut = iy_arr.mutable_unchecked<1>();
    auto res_mut = res_arr.mutable_unchecked<1>();
    auto count_mut = count_arr.mutable_unchecked<1>();
    auto mean_z_mut = mean_z_arr.mutable_unchecked<1>();
    auto min_z_mut = min_z_arr.mutable_unchecked<1>();
    auto max_z_mut = max_z_arr.mutable_unchecked<1>();
    auto class_mut = class_arr.mutable_unchecked<1>();
    auto conf_mut = conf_arr.mutable_unchecked<1>();
    auto trav_mut = trav_arr.mutable_unchecked<1>();

    for (size_t i = 0; i < M; ++i) {
        const auto& c = cells[i];
        bands_list.push_back(c.band_name);
        ix_mut(i) = c.ix;
        iy_mut(i) = c.iy;
        res_mut(i) = c.resolution;
        count_mut(i) = c.point_count;
        mean_z_mut(i) = c.elevation_mean;
        min_z_mut(i) = c.elevation_min;
        max_z_mut(i) = c.elevation_max;
        class_mut(i) = static_cast<int64_t>(c.semantic_class);
        conf_mut(i) = c.confidence;
        trav_mut(i) = c.traversability;
    }

    py::dict out;
    out["bands"] = py::cast(bands_list);
    out["ix"] = ix_arr;
    out["iy"] = iy_arr;
    out["resolution"] = res_arr;
    out["point_count"] = count_arr;
    out["elevation_mean"] = mean_z_arr;
    out["elevation_min"] = min_z_arr;
    out["elevation_max"] = max_z_arr;
    out["semantic_class"] = class_arr;
    out["confidence"] = conf_arr;
    out["traversability"] = trav_arr;
    out["num_cells"] = M;
    return out;
}

} // anonymous namespace

PYBIND11_MODULE(foveated_grid_cpp, m) {
    m.doc() = "C++ Foveated 2.5D Spatial Grid Engine Python Extension (pybind11)";

    // SuperClass enum
    py::enum_<SuperClass>(m, "SuperClass")
        .value("DRIVABLE_TERRAIN", SuperClass::DRIVABLE_TERRAIN)
        .value("NON_DRIVABLE_TERRAIN", SuperClass::NON_DRIVABLE_TERRAIN)
        .value("STATIC_OBSTACLE", SuperClass::STATIC_OBSTACLE)
        .value("DYNAMIC_OBJECT", SuperClass::DYNAMIC_OBJECT)
        .value("IGNORE_LABEL", SuperClass::IGNORE_LABEL)
        .export_values();

    // ClassifiedPoint struct
    py::class_<ClassifiedPoint>(m, "ClassifiedPoint")
        .def(py::init<>())
        .def(py::init<float, float, float, float, uint8_t, float>(),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("intensity") = 0.0f,
             py::arg("class_id") = SuperClass::IGNORE_LABEL, py::arg("confidence") = 1.0f)
        .def_readwrite("x", &ClassifiedPoint::x)
        .def_readwrite("y", &ClassifiedPoint::y)
        .def_readwrite("z", &ClassifiedPoint::z)
        .def_readwrite("intensity", &ClassifiedPoint::intensity)
        .def_readwrite("class_id", &ClassifiedPoint::class_id)
        .def_readwrite("confidence", &ClassifiedPoint::confidence)
        .def("__repr__", [](const ClassifiedPoint& p) {
            return "<ClassifiedPoint (" + std::to_string(p.x) + ", " + std::to_string(p.y) + ", " + std::to_string(p.z) + ")>";
        });

    // FoveationBand struct
    py::class_<FoveationBand>(m, "FoveationBand")
        .def(py::init<std::string, float, float, float>(),
             py::arg("name"), py::arg("min_range"), py::arg("max_range"), py::arg("voxel_size"))
        .def_readwrite("name", &FoveationBand::name)
        .def_readwrite("min_range", &FoveationBand::min_range)
        .def_readwrite("max_range", &FoveationBand::max_range)
        .def_readwrite("voxel_size", &FoveationBand::voxel_size)
        .def("contains", &FoveationBand::contains);

    // GridCell struct
    py::class_<GridCell>(m, "GridCell")
        .def(py::init<>())
        .def_readwrite("band_name", &GridCell::band_name)
        .def_readwrite("ix", &GridCell::ix)
        .def_readwrite("iy", &GridCell::iy)
        .def_readwrite("resolution", &GridCell::resolution)
        .def_readwrite("point_count", &GridCell::point_count)
        .def_readwrite("elevation_mean", &GridCell::elevation_mean)
        .def_readwrite("elevation_min", &GridCell::elevation_min)
        .def_readwrite("elevation_max", &GridCell::elevation_max)
        .def_readwrite("semantic_class", &GridCell::semantic_class)
        .def_readwrite("confidence", &GridCell::confidence)
        .def_readwrite("traversability", &GridCell::traversability)
        .def("min_x", &GridCell::min_x)
        .def("max_x", &GridCell::max_x)
        .def("min_y", &GridCell::min_y)
        .def("max_y", &GridCell::max_y)
        .def("height_range", &GridCell::height_range);


    // FoveatedGridEngine class
    py::class_<FoveatedGridEngine>(m, "FoveatedGridEngine")
        .def(py::init<>())
        .def(py::init<const std::vector<FoveationBand>&>(), py::arg("custom_bands"))
        .def("build_grid", &FoveatedGridEngine::build_grid, py::arg("points"),
             "Builds grid from a vector of ClassifiedPoint objects")
        .def("build_grid_numpy", &build_grid_numpy_impl,
             py::arg("points"), py::arg("labels") = py::none(), py::arg("confidences") = py::none(),
             "Fast vectorized grid generation directly consuming NumPy array buffers")
        .def("resolve_band", [](const FoveatedGridEngine& self, float r) -> std::optional<FoveationBand> {
            const auto* b = self.resolve_band(r);
            if (b) return *b;
            return std::nullopt;
        }, py::arg("r"))
        .def_static("xy_to_cell", &FoveatedGridEngine::xy_to_cell, py::arg("x"), py::arg("y"), py::arg("resolution"))
        .def_static("load_points_csv", &FoveatedGridEngine::load_points_csv, py::arg("filepath"))
        .def_static("export_grid_csv", &FoveatedGridEngine::export_grid_csv, py::arg("cells"), py::arg("filepath"));
}
