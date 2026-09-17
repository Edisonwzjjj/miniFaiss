cmake_minimum_required(VERSION 3.25)

project(minifaiss LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(enable_project_warnings target)
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Werror
    )
endfunction()

add_library(minifaiss STATIC
    src/index_flat_ip.cpp
    src/index_ivf_flat.cpp
    src/index_ivf_pq.cpp
    src/product_quantizer.cpp
    src/search_executor.cpp
)

target_include_directories(minifaiss
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_project_warnings(minifaiss)

add_executable(minifaiss_demo
    examples/search_demo.cpp
)
target_link_libraries(minifaiss_demo PRIVATE minifaiss)
enable_project_warnings(minifaiss_demo)

add_executable(minifaiss_tests
    tests/index_flat_ip_test.cpp
)
target_link_libraries(minifaiss_tests PRIVATE minifaiss)
enable_project_warnings(minifaiss_tests)

enable_testing()
add_test(NAME minifaiss_tests COMMAND minifaiss_tests)

add_executable(minifaiss_ivf_flat_tests
    tests/index_ivf_flat_test.cpp
)
target_link_libraries(minifaiss_ivf_flat_tests PRIVATE minifaiss)
enable_project_warnings(minifaiss_ivf_flat_tests)
add_test(NAME minifaiss_ivf_flat_tests COMMAND minifaiss_ivf_flat_tests)

add_executable(minifaiss_ivf_pq_tests
    tests/index_ivf_pq_test.cpp
)
target_link_libraries(minifaiss_ivf_pq_tests PRIVATE minifaiss)
enable_project_warnings(minifaiss_ivf_pq_tests)
add_test(NAME minifaiss_ivf_pq_tests COMMAND minifaiss_ivf_pq_tests)

add_executable(minifaiss_product_quantizer_tests
    tests/product_quantizer_test.cpp
)
target_link_libraries(minifaiss_product_quantizer_tests PRIVATE minifaiss)
enable_project_warnings(minifaiss_product_quantizer_tests)
add_test(
    NAME minifaiss_product_quantizer_tests
    COMMAND minifaiss_product_quantizer_tests
)

add_executable(minifaiss_search_executor_tests
    tests/search_executor_test.cpp
)
target_link_libraries(minifaiss_search_executor_tests PRIVATE minifaiss)
enable_project_warnings(minifaiss_search_executor_tests)
add_test(NAME minifaiss_search_executor_tests
         COMMAND minifaiss_search_executor_tests)

add_executable(minifaiss_bench
    benchmarks/bench_flat_ip.cpp
)
target_link_libraries(minifaiss_bench PRIVATE minifaiss)
enable_project_warnings(minifaiss_bench)

add_executable(minifaiss_dot_product_bench
    benchmarks/bench_dot_product.cpp
)
target_link_libraries(minifaiss_dot_product_bench PRIVATE minifaiss)
enable_project_warnings(minifaiss_dot_product_bench)
