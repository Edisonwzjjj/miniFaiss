#include "minifaiss/index_flat_ip.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <span>
#include <vector>

int main() {
    constexpr std::size_t item_count = 10'000;
    constexpr std::size_t dimension = 128;
    constexpr std::size_t query_count = 100;
    constexpr std::size_t k = 10;

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> items(item_count * dimension);
    for (float& value : items) {
        value = distribution(generator);
    }

    minifaiss::IndexFlatIP index(dimension);
    index.add(items);

    std::vector<float> queries(query_count * dimension);
    for (float& value : queries) {
        value = distribution(generator);
    }

    (void)index.search(std::span<const float>(queries.data(), dimension), k);

    std::size_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();

    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const float* query_data = queries.data() + query_id * dimension;
        const std::span<const float> query(query_data, dimension);
        const auto results = index.search(query, k);

        for (const minifaiss::SearchResult& result : results) {
            checksum += result.id;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double queries_per_second =
        static_cast<double>(query_count) / elapsed.count();

    std::cout << "N=" << item_count << ", D=" << dimension
              << ", Q=" << query_count << ", K=" << k << '\n';
    std::cout << "elapsed_seconds=" << elapsed.count() << '\n';
    std::cout << "queries_per_second=" << queries_per_second << '\n';
    std::cout << "checksum=" << checksum << '\n';

    return 0;
}
