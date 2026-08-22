cmake_minimum_required(VERSION 3.14)
project(foveated_grid_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization and warning flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

include_directories(include)

# Static Library
add_library(foveated_grid_lib STATIC
    src/foveated_grid.cpp
)

# CLI Binary
add_executable(foveated_grid_cli
    src/main.cpp
)
target_link_libraries(foveated_grid_cli PRIVATE foveated_grid_lib)

# Standalone C++ Unit Test Executable
add_executable(foveated_grid_tests
    tests/test_grid.cpp
)
target_link_libraries(foveated_grid_tests PRIVATE foveated_grid_lib)
