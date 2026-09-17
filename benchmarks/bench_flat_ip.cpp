#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "minifaiss/index_flat_ip.hpp"
#include "minifaiss/search_executor.hpp"

namespace {

using BatchResults = std::vector<std::vector<minifaiss::SearchResult>>;

bool same_results(const BatchResults& lhs, const BatchResults& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (std::size_t query_id = 0; query_id < lhs.size(); ++query_id) {
        if (lhs[query_id].size() != rhs[query_id].size()) {
            return false;
        }

        for (std::size_t rank = 0; rank < lhs[query_id].size(); ++rank) {
            if (lhs[query_id][rank].id != rhs[query_id][rank].id ||
                lhs[query_id][rank].score != rhs[query_id][rank].score) {
                return false;
            }
        }
    }

    return true;
}

std::uint64_t fingerprint(const BatchResults& results) {
    std::uint64_t hash = 1469598103934665603ULL;

    for (std::size_t query_id = 0; query_id < results.size(); ++query_id) {
        for (std::size_t rank = 0; rank < results[query_id].size(); ++rank) {
            const minifaiss::SearchResult& result = results[query_id][rank];
            const std::uint64_t score_bits =
                std::bit_cast<std::uint32_t>(result.score);

            hash ^= query_id;
            hash *= 1099511628211ULL;
            hash ^= rank;
            hash *= 1099511628211ULL;
            hash ^= result.id;
            hash *= 1099511628211ULL;
            hash ^= score_bits;
            hash *= 1099511628211ULL;
        }
    }

    return hash;
}

BatchResults scalar_search(const minifaiss::IndexFlatIP& index,
                           std::span<const float> queries, std::size_t k) {
    const std::size_t query_count = queries.size() / index.dimension();
    BatchResults results;
    results.reserve(query_count);

    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const std::span<const float> query(
            queries.data() + query_id * index.dimension(), index.dimension());
        results.push_back(index.search(query, k));
    }

    return results;
}

BatchResults executor_scalar_search(minifaiss::SearchExecutor& executor,
                                    const minifaiss::IndexFlatIP& index,
                                    std::span<const float> queries,
                                    std::size_t k) {
    const std::size_t query_count = queries.size() / index.dimension();
    std::vector<std::future<std::vector<minifaiss::SearchResult>>> futures;
    futures.reserve(query_count);

    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const std::span<const float> query(
            queries.data() + query_id * index.dimension(), index.dimension());
        futures.push_back(executor.submit(
            [&index, query, k] { return index.search(query, k); }));
    }

    BatchResults results(query_count);
    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        results[query_id] = futures[query_id].get();
    }

    return results;
}

BatchResults executor_chunked_batch_search(minifaiss::SearchExecutor& executor,
                                           const minifaiss::IndexFlatIP& index,
                                           std::span<const float> queries,
                                           std::size_t k,
                                           std::size_t outer_workers) {
    const std::size_t query_count = queries.size() / index.dimension();
    std::vector<std::future<BatchResults>> futures;
    futures.reserve(outer_workers);

    for (std::size_t worker_id = 0; worker_id < outer_workers; ++worker_id) {
        const std::size_t begin = query_count * worker_id / outer_workers;
        const std::size_t end = query_count * (worker_id + 1) / outer_workers;
        const std::span<const float> chunk(
            queries.data() + begin * index.dimension(),
            (end - begin) * index.dimension());

        futures.push_back(executor.submit(
            [&index, chunk, k] { return index.search_batch(chunk, k); }));
    }

    BatchResults results;
    results.reserve(query_count);
    for (auto& future : futures) {
        BatchResults chunk_results = future.get();
        results.insert(results.end(),
                       std::make_move_iterator(chunk_results.begin()),
                       std::make_move_iterator(chunk_results.end()));
    }

    return results;
}

template <typename Function>
void benchmark_mode(std::string_view mode, std::size_t query_count,
                    std::size_t outer_workers, std::size_t chunk_queries,
                    const BatchResults& expected, Function&& function) {
    constexpr std::size_t kWarmups = 2;
    constexpr std::size_t kRepetitions = 5;

    for (std::size_t warmup = 0; warmup < kWarmups; ++warmup) {
        const BatchResults results = function();
        if (!same_results(results, expected)) {
            throw std::runtime_error(
                "warmup results differ from scalar reference");
        }
    }

    std::array<double, kRepetitions> samples = {};
    std::uint64_t result_fingerprint = 0;
    for (std::size_t repeat = 0; repeat < kRepetitions; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        const BatchResults results = function();
        const auto end = std::chrono::steady_clock::now();

        if (!same_results(results, expected)) {
            throw std::runtime_error(
                "timed results differ from scalar reference");
        }

        samples[repeat] = std::chrono::duration<double>(end - start).count();
        result_fingerprint = fingerprint(results);
    }

    std::sort(samples.begin(), samples.end());
    const double median_seconds = samples[kRepetitions / 2];
    const double queries_per_second =
        static_cast<double>(query_count) / median_seconds;

    std::cout << "mode=" << mode << " Q=" << query_count
              << " outer_workers=" << outer_workers
              << " chunk_queries=" << chunk_queries << " warmups=" << kWarmups
              << " repeats=" << kRepetitions
              << " median_seconds=" << median_seconds
              << " queries_per_second=" << queries_per_second
              << " result_fingerprint=0x" << std::hex << result_fingerprint
              << std::dec << " status=ok\n";
}

}  // namespace

int main() {
    constexpr std::size_t kItemCount = 10'000;
    constexpr std::size_t kDimension = 128;
    constexpr std::size_t kQueryCount = 1'024;
    constexpr std::size_t k = 10;
    constexpr std::size_t kOuterWorkers = 4;
    constexpr unsigned int kSeed = 42;

    std::mt19937 generator(kSeed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> items(kItemCount * kDimension);
    for (float& value : items) {
        value = distribution(generator);
    }

    std::vector<float> queries(kQueryCount * kDimension);
    for (float& value : queries) {
        value = distribution(generator);
    }

    minifaiss::IndexFlatIP index(kDimension);
    index.add(items);

    const BatchResults expected = scalar_search(index, queries, k);

    std::cout << "index=flat_ip N=" << kItemCount << " D=" << kDimension
              << " Q=" << kQueryCount << " K=" << k
              << " outer_workers=" << kOuterWorkers << " seed=" << kSeed
              << "\n";
    std::cout << "nested_executor_chunked_batch may oversubscribe CPUs: "
              << "outer workers each invoke inner batch parallelism.\n";

    benchmark_mode("scalar_serial", kQueryCount, 0, 0, expected,
                   [&] { return scalar_search(index, queries, k); });
    benchmark_mode("inner_batch_parallel", kQueryCount, 0, kQueryCount,
                   expected, [&] { return index.search_batch(queries, k); });

    minifaiss::SearchExecutor outer_executor(kOuterWorkers, kQueryCount);
    benchmark_mode(
        "outer_executor_scalar", kQueryCount, kOuterWorkers, 1, expected, [&] {
            return executor_scalar_search(outer_executor, index, queries, k);
        });

    minifaiss::SearchExecutor nested_executor(kOuterWorkers, kOuterWorkers);
    benchmark_mode("nested_executor_chunked_batch", kQueryCount, kOuterWorkers,
                   kQueryCount / kOuterWorkers, expected, [&] {
                       return executor_chunked_batch_search(
                           nested_executor, index, queries, k, kOuterWorkers);
                   });

    return 0;
}
