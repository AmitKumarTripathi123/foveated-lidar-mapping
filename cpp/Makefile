CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -Iinclude
BIN_DIR = bin

all: $(BIN_DIR)/foveated_grid_cli $(BIN_DIR)/foveated_grid_tests

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/foveated_grid_cli: src/foveated_grid.cpp src/main.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/foveated_grid_tests: src/foveated_grid.cpp tests/test_grid.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(BIN_DIR)/foveated_grid_tests
	./$(BIN_DIR)/foveated_grid_tests

clean:
	rm -rf $(BIN_DIR)

.PHONY: all test clean
