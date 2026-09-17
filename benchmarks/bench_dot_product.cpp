#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "minifaiss/detail/dot_product.hpp"

#if defined(__APPLE__)
#include <os/signpost.h>
#endif

namespace {

struct BenchmarkConfig {
    std::vector<std::size_t> dimensions = {7, 128, 129};
    std::size_t vector_pairs = 20'000;
    std::size_t duration_seconds = 5;
    unsigned int seed = 42;
};

struct WorkloadResult {
    double elapsed_seconds;
    std::size_t passes;
    std::size_t dot_products;
    float checksum;
};

#if defined(__clang__)
#define MINIFAISS_NOINLINE __attribute__((noinline))
#else
#define MINIFAISS_NOINLINE
#endif

MINIFAISS_NOINLINE float scalar_dot_product(const float* lhs, const float* rhs,
                                            std::size_t length) {
    volatile const float* left = lhs;
    volatile const float* right = rhs;
    float sum = 0.0F;

    for (std::size_t value_id = 0; value_id < length; ++value_id) {
        sum += left[value_id] * right[value_id];
    }

    return sum;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--duration-seconds N] [--vector-pairs N]"
              << " [--dimensions N,N,...]\n";
}

bool parse_positive_size(std::string_view text, std::size_t* value) {
    if (text.empty()) {
        return false;
    }

    std::size_t parsed = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed == 0) {
        return false;
    }

    *value = parsed;
    return true;
}

bool parse_dimensions(std::string_view text,
                      std::vector<std::size_t>* dimensions) {
    std::vector<std::size_t> parsed_dimensions;
    std::size_t begin = 0;

    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::size_t length =
            end == std::string_view::npos ? text.size() - begin : end - begin;
        std::size_t dimension = 0;
        if (!parse_positive_size(text.substr(begin, length), &dimension)) {
            return false;
        }
        parsed_dimensions.push_back(dimension);

        if (end == std::string_view::npos) {
            break;
        }
        if (end + 1 == text.size()) {
            return false;
        }
        begin = end + 1;
    }

    if (parsed_dimensions.empty()) {
        return false;
    }

    *dimensions = std::move(parsed_dimensions);
    return true;
}

bool parse_arguments(int argc, char* argv[], BenchmarkConfig* config) {
    for (int argument_id = 1; argument_id < argc; ++argument_id) {
        const std::string_view argument(argv[argument_id]);
        if (argument == "--help") {
            return false;
        }

        if (argument_id + 1 >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return false;
        }

        const std::string_view value(argv[++argument_id]);
        if (argument == "--duration-seconds") {
            if (!parse_positive_size(value, &config->duration_seconds)) {
                std::cerr << "Invalid duration seconds: " << value << '\n';
                return false;
            }
        } else if (argument == "--vector-pairs") {
            if (!parse_positive_size(value, &config->vector_pairs)) {
                std::cerr << "Invalid vector pairs: " << value << '\n';
                return false;
            }
        } else if (argument == "--dimensions") {
            if (!parse_dimensions(value, &config->dimensions)) {
                std::cerr << "Invalid dimensions: " << value << '\n';
                return false;
            }
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            return false;
        }
    }

    return true;
}

bool can_allocate(std::size_t vector_pairs, std::size_t dimension) {
    return vector_pairs <= std::numeric_limits<std::size_t>::max() / dimension;
}

enum class BenchmarkMode {
    kScalar,
    kSimd,
};

#if defined(__APPLE__)
os_log_t signpost_log() {
    static os_log_t log =
        os_log_create("com.zijunwang.minifaiss", "DotProduct");
    return log;
}

void begin_signpost(BenchmarkMode mode, std::size_t dimension,
                    const BenchmarkConfig& config) {
    const os_log_t log = signpost_log();
    if (!os_signpost_enabled(log)) {
        return;
    }

    if (mode == BenchmarkMode::kScalar) {
        os_signpost_interval_begin(
            log, OS_SIGNPOST_ID_EXCLUSIVE, "Scalar",
            "dimension=%{public}zu vector_pairs=%{public}zu "
            "duration_seconds=%{public}zu",
            dimension, config.vector_pairs, config.duration_seconds);
    } else {
        os_signpost_interval_begin(
            log, OS_SIGNPOST_ID_EXCLUSIVE, "SIMD",
            "dimension=%{public}zu vector_pairs=%{public}zu "
            "duration_seconds=%{public}zu",
            dimension, config.vector_pairs, config.duration_seconds);
    }
}

void end_signpost(BenchmarkMode mode, std::size_t dimension,
                  const WorkloadResult& result) {
    const os_log_t log = signpost_log();
    if (!os_signpost_enabled(log)) {
        return;
    }

    if (mode == BenchmarkMode::kScalar) {
        os_signpost_interval_end(
            log, OS_SIGNPOST_ID_EXCLUSIVE, "Scalar",
            "dimension=%{public}zu passes=%{public}zu dot_products=%{public}zu",
            dimension, result.passes, result.dot_products);
    } else {
        os_signpost_interval_end(
            log, OS_SIGNPOST_ID_EXCLUSIVE, "SIMD",
            "dimension=%{public}zu passes=%{public}zu dot_products=%{public}zu",
            dimension, result.passes, result.dot_products);
    }
}
#else
void begin_signpost(BenchmarkMode, std::size_t, const BenchmarkConfig&) {}
void end_signpost(BenchmarkMode, std::size_t, const WorkloadResult&) {}
#endif

template <typename DotProduct>
WorkloadResult run_workload(DotProduct&& dot_product,
                            std::span<const float> lhs,
                            std::span<const float> rhs, std::size_t dimension,
                            const BenchmarkConfig& config) {
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(config.duration_seconds);
    float checksum = 0.0F;
    std::size_t passes = 0;

    do {
        for (std::size_t pair_id = 0; pair_id < config.vector_pairs;
             ++pair_id) {
            const std::size_t shifted_id =
                (pair_id + passes) % config.vector_pairs;
            checksum +=
                dot_product(lhs.data() + shifted_id * dimension,
                            rhs.data() + shifted_id * dimension, dimension);
        }
        ++passes;
    } while (std::chrono::steady_clock::now() < deadline);

    const auto end = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double>(end - start).count(),
        passes,
        passes * config.vector_pairs,
        checksum,
    };
}

bool close_enough(float lhs, float rhs) {
    const float difference = std::abs(lhs - rhs);
    const float scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
    return difference <= 1e-5F * scale;
}

bool verify_dot_products(std::span<const float> lhs, std::span<const float> rhs,
                         std::size_t dimension, std::size_t vector_pairs) {
    for (std::size_t pair_id = 0; pair_id < vector_pairs; ++pair_id) {
        const float scalar =
            scalar_dot_product(lhs.data() + pair_id * dimension,
                               rhs.data() + pair_id * dimension, dimension);
        const float simd = minifaiss::detail::dot_product(
            lhs.data() + pair_id * dimension, rhs.data() + pair_id * dimension,
            dimension);

        if (!close_enough(scalar, simd)) {
            return false;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    if (!parse_arguments(argc, argv, &config)) {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            return EXIT_SUCCESS;
        }
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::mt19937 generator(config.seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::cout << "benchmark=dot_product vector_pairs=" << config.vector_pairs
              << " duration_seconds=" << config.duration_seconds
              << " seed=" << config.seed << " dimensions=";
    for (std::size_t dimension_id = 0; dimension_id < config.dimensions.size();
         ++dimension_id) {
        if (dimension_id != 0) {
            std::cout << ',';
        }
        std::cout << config.dimensions[dimension_id];
    }
    std::cout << '\n';

    for (const std::size_t dimension : config.dimensions) {
        if (!can_allocate(config.vector_pairs, dimension)) {
            std::cerr << "vector-pairs times dimension overflows size_t\n";
            return EXIT_FAILURE;
        }

        std::vector<float> lhs(config.vector_pairs * dimension);
        std::vector<float> rhs(config.vector_pairs * dimension);
        for (float& value : lhs) {
            value = distribution(generator);
        }
        for (float& value : rhs) {
            value = distribution(generator);
        }

        if (!verify_dot_products(lhs, rhs, dimension, config.vector_pairs)) {
            std::cerr << "Scalar and SIMD results differ for dimension "
                      << dimension << '\n';
            return EXIT_FAILURE;
        }

        begin_signpost(BenchmarkMode::kScalar, dimension, config);
        const WorkloadResult scalar =
            run_workload(scalar_dot_product, lhs, rhs, dimension, config);
        end_signpost(BenchmarkMode::kScalar, dimension, scalar);

        begin_signpost(BenchmarkMode::kSimd, dimension, config);
        const WorkloadResult simd = run_workload(minifaiss::detail::dot_product,
                                                 lhs, rhs, dimension, config);
        end_signpost(BenchmarkMode::kSimd, dimension, simd);

        const double scalar_dots_per_second =
            static_cast<double>(scalar.dot_products) / scalar.elapsed_seconds;
        const double simd_dots_per_second =
            static_cast<double>(simd.dot_products) / simd.elapsed_seconds;
        const double speedup = simd_dots_per_second / scalar_dots_per_second;

        std::cout << "dimension=" << dimension
                  << " mode=scalar elapsed_seconds=" << scalar.elapsed_seconds
                  << " passes=" << scalar.passes
                  << " dot_products=" << scalar.dot_products
                  << " dots_per_second=" << scalar_dots_per_second
                  << " checksum=" << scalar.checksum << '\n';
        std::cout << "dimension=" << dimension
                  << " mode=simd elapsed_seconds=" << simd.elapsed_seconds
                  << " passes=" << simd.passes
                  << " dot_products=" << simd.dot_products
                  << " dots_per_second=" << simd_dots_per_second
                  << " checksum=" << simd.checksum << '\n';
        std::cout << "dimension=" << dimension << " speedup=" << speedup
                  << '\n';
    }

    return EXIT_SUCCESS;
}
