BUILD_DIR := .build
BUILD_TYPE ?= Debug

.PHONY: configure build check test run bench clean

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR)

check: build

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	$(BUILD_DIR)/minifaiss_demo

bench:
	$(MAKE) BUILD_TYPE=Release build
	$(BUILD_DIR)/minifaiss_bench

clean:
	cmake -E rm -rf $(BUILD_DIR)
