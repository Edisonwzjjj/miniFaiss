#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "minifaiss/index_flat_ip.hpp"
#include "minifaiss/index_ivf_flat.hpp"

namespace {

std::size_t count_matching_ids(
    const std::vector<minifaiss::SearchResult>& expected,
    const std::vector<minifaiss::SearchResult>& actual) {
    std::size_t matches = 0;

    for (const minifaiss::SearchResult& actual_result : actual) {
        for (const minifaiss::SearchResult& expected_result : expected) {
            if (actual_result.id == expected_result.id) {
                ++matches;
                break;
            }
        }
    }

    return matches;
}

}  // namespace

int main() {
    constexpr std::size_t item_count = 10'000;
    constexpr std::size_t dimension = 128;
    constexpr std::size_t query_count = 100;
    constexpr std::size_t k = 10;
    constexpr std::size_t nlist = 100;
    constexpr std::size_t train_iterations = 10;
    constexpr unsigned int seed = 42;
    constexpr std::array<std::size_t, 8> nprobe_values = {
        1, 2, 4, 8, 16, 32, 64, 100,
    };

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> items(item_count * dimension);
    for (float& value : items) {
        value = distribution(generator);
    }

    std::vector<float> queries(query_count * dimension);
    for (float& value : queries) {
        value = distribution(generator);
    }

    minifaiss::IndexFlatIP flat_index(dimension);
    flat_index.add(items);

    const auto flat_results = flat_index.search_batch(queries, k);
    std::size_t ground_truth_result_count = 0;
    std::size_t flat_checksum = 0;
    for (const auto& results : flat_results) {
        ground_truth_result_count += results.size();
        for (const minifaiss::SearchResult& result : results) {
            flat_checksum += result.id;
        }
    }

    (void)flat_index.search_batch(queries, k);

    const auto flat_start = std::chrono::steady_clock::now();
    const auto timed_flat_results = flat_index.search_batch(queries, k);
    const auto flat_end = std::chrono::steady_clock::now();
    std::size_t timed_flat_checksum = 0;
    for (const auto& results : timed_flat_results) {
        for (const minifaiss::SearchResult& result : results) {
            timed_flat_checksum += result.id;
        }
    }
    const std::chrono::duration<double> flat_elapsed = flat_end - flat_start;
    const double flat_queries_per_second =
        static_cast<double>(query_count) / flat_elapsed.count();

    minifaiss::IndexIVFFlat ivf_index(dimension, nlist);
    const auto train_start = std::chrono::steady_clock::now();
    ivf_index.train(items, train_iterations);
    const auto train_end = std::chrono::steady_clock::now();
    ivf_index.add(items);
    const std::chrono::duration<double> train_elapsed = train_end - train_start;

    std::cout << "N=" << item_count << " D=" << dimension
              << " Q=" << query_count << " K=" << k << " nlist=" << nlist
              << " train_iterations=" << train_iterations << " seed=" << seed
              << '\n';
    std::cout << "index=flat_ip batch_elapsed_seconds=" << flat_elapsed.count()
              << " queries_per_second=" << flat_queries_per_second
              << " checksum=" << timed_flat_checksum << '\n';
    std::cout << "ground_truth_checksum=" << flat_checksum
              << " ground_truth_result_count=" << ground_truth_result_count
              << '\n';
    std::cout << "ivf_train_seconds=" << train_elapsed.count() << '\n';

    for (const std::size_t nprobe : nprobe_values) {
        (void)ivf_index.search_batch(queries, k, nprobe);

        const auto start = std::chrono::steady_clock::now();
        const auto batch_results = ivf_index.search_batch(queries, k, nprobe);
        const auto end = std::chrono::steady_clock::now();

        std::size_t checksum = 0;
        std::size_t matching_ids = 0;
        for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
            const auto& results = batch_results[query_id];
            for (const minifaiss::SearchResult& result : results) {
                checksum += result.id;
            }
            matching_ids += count_matching_ids(flat_results[query_id], results);
        }

        const std::chrono::duration<double> elapsed = end - start;
        const double queries_per_second =
            static_cast<double>(query_count) / elapsed.count();
        const double recall_at_k =
            ground_truth_result_count == 0
                ? 0.0
                : static_cast<double>(matching_ids) /
                      static_cast<double>(ground_truth_result_count);

        std::cout << "index=ivf_flat nprobe=" << nprobe
                  << " batch_elapsed_seconds=" << elapsed.count()
                  << " queries_per_second=" << queries_per_second
                  << " recall_at_k=" << recall_at_k << " checksum=" << checksum
                  << '\n';
    }

    return 0;
}
