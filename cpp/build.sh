#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/bin"
CXX="${CXX:-clang++}"
echo "Compiling C++ Foveated Grid Engine using ${CXX}..."
${CXX} -std=c++17 -O3 -Wall -Wextra -I"${SCRIPT_DIR}/include" \
    "${SCRIPT_DIR}/src/foveated_grid.cpp" "${SCRIPT_DIR}/src/main.cpp" \
    -o "${SCRIPT_DIR}/bin/foveated_grid_cli"

${CXX} -std=c++17 -O3 -Wall -Wextra -I"${SCRIPT_DIR}/include" \
    "${SCRIPT_DIR}/src/foveated_grid.cpp" "${SCRIPT_DIR}/tests/test_grid.cpp" \
    -o "${SCRIPT_DIR}/bin/foveated_grid_tests"

echo "Build complete: ${SCRIPT_DIR}/bin/foveated_grid_cli"
echo "Running C++ test suite..."
"${SCRIPT_DIR}/bin/foveated_grid_tests"
