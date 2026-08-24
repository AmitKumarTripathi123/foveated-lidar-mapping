cmake_minimum_required(VERSION 3.14)
project(foveated_grid_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /Oi /Ot /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

include_directories(include)

find_package(pybind11 REQUIRED)

pybind11_add_module(foveated_grid_cpp
    src/grid_accumulator.cpp
    src/grid_rasterizer.cpp
    src/bindings.cpp
)
