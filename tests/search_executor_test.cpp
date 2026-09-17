#include "minifaiss/search_executor.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

#include "minifaiss/index_flat_ip.hpp"
#include "minifaiss/index_ivf_flat.hpp"
#include "minifaiss/index_ivf_pq.hpp"

namespace {

class TestRunner {
public:
    void check(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures_;
        }
    }

    template <typename Function>
    void check_throws_runtime_error(Function&& function, const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::runtime_error&) {
        }
    }

    [[nodiscard]] int exit_code() const { return failures_ == 0 ? 0 : 1; }

private:
    int failures_ = 0;
};

bool same_batch_results(
    const std::vector<std::vector<minifaiss::SearchResult>>& lhs,
    const std::vector<std::vector<minifaiss::SearchResult>>& rhs) {
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

void test_tasks_return_values_and_propagate_exceptions(TestRunner& tests) {
    minifaiss::SearchExecutor executor(2, 2);

    auto first = executor.submit([] { return 7; });
    auto second = executor.submit([] { return 11; });
    auto failure = executor.submit(
        []() -> int { throw std::runtime_error("expected task failure"); });

    tests.check(first.get() == 7,
                "first task result is returned through its future");
    tests.check(second.get() == 11,
                "second task result is returned through its future");
    tests.check_throws_runtime_error(
        [&failure] { (void)failure.get(); },
        "task exception is returned through its future");

    auto later_task = executor.submit([] { return 13; });
    tests.check(later_task.get() == 13,
                "worker continues after a task throws an exception");
}

void test_shutdown_drains_tasks_and_rejects_submissions(TestRunner& tests) {
    minifaiss::SearchExecutor executor(1, 1);

    auto first = executor.submit([] { return 3; });
    auto second = executor.submit([] { return 5; });
    executor.shutdown();

    tests.check(first.get() == 3, "shutdown drains the first accepted task");
    tests.check(second.get() == 5, "shutdown drains queued accepted tasks");
    tests.check_throws_runtime_error(
        [&executor] { (void)executor.submit([] { return 7; }); },
        "shutdown rejects new tasks");
}

void test_parallel_batch_search_matches_scalar_results(TestRunner& tests) {
    minifaiss::IndexFlatIP index(2);
    index.add(std::array<float, 8>{
        1.0F,
        0.0F,
        0.0F,
        1.0F,
        2.0F,
        2.0F,
        -1.0F,
        1.0F,
    });

    const std::array<float, 16> queries = {
        1.0F, 0.0F, 0.0F, 1.0F, 1.0F,  1.0F, -1.0F, 1.0F,
        2.0F, 0.0F, 0.0F, 2.0F, -1.0F, 0.0F, 0.0F,  -1.0F,
    };

    std::vector<std::vector<minifaiss::SearchResult>> expected;
    expected.reserve(8);
    for (std::size_t query_id = 0; query_id < 8; ++query_id) {
        const std::span<const float> query(queries.data() + query_id * 2, 2);
        expected.push_back(index.search(query, 3));
    }

    const auto first_batch = index.search_batch(queries, 3);
    const auto second_batch = index.search_batch(queries, 3);
    tests.check(same_batch_results(first_batch, expected),
                "parallel batch search matches scalar results");
    tests.check(same_batch_results(second_batch, expected),
                "parallel batch search is repeatable");
}

void test_nested_executor_and_batch_parallelism(TestRunner& tests) {
    minifaiss::IndexFlatIP index(2);
    index.add(std::array<float, 8>{
        1.0F,
        0.0F,
        0.0F,
        1.0F,
        2.0F,
        2.0F,
        -1.0F,
        1.0F,
    });

    const std::array<float, 32> queries = {
        1.0F,  0.0F,  0.0F, 1.0F, 1.0F,  1.0F, -1.0F, 1.0F, 2.0F, 0.0F,  0.0F,
        2.0F,  -1.0F, 0.0F, 0.0F, -1.0F, 1.0F, 0.0F,  0.0F, 1.0F, 1.0F,  1.0F,
        -1.0F, 1.0F,  2.0F, 0.0F, 0.0F,  2.0F, -1.0F, 0.0F, 0.0F, -1.0F,
    };

    std::vector<std::vector<minifaiss::SearchResult>> expected;
    expected.reserve(16);
    for (std::size_t query_id = 0; query_id < 16; ++query_id) {
        const std::span<const float> query(queries.data() + query_id * 2, 2);
        expected.push_back(index.search(query, 3));
    }

    minifaiss::SearchExecutor executor(2, 2);
    const std::span<const float> first_chunk(queries.data(), 16);
    const std::span<const float> second_chunk(queries.data() + 16, 16);
    auto first_future =
        executor.submit([&] { return index.search_batch(first_chunk, 3); });
    auto second_future =
        executor.submit([&] { return index.search_batch(second_chunk, 3); });

    std::vector<std::vector<minifaiss::SearchResult>> actual =
        first_future.get();
    const auto second_results = second_future.get();
    actual.insert(actual.end(), second_results.begin(), second_results.end());

    tests.check(same_batch_results(actual, expected),
                "nested executor and batch parallelism preserves query order");
}

void test_async_batch_search_matches_synchronous_results(TestRunner& tests) {
    minifaiss::IndexFlatIP flat_index(2);
    flat_index.add(std::array<float, 6>{
        1.0F,
        0.0F,
        0.0F,
        1.0F,
        2.0F,
        2.0F,
    });

    minifaiss::IndexIVFFlat ivf_index(2, 2);
    ivf_index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    ivf_index.add(std::array<float, 6>{
        0.9F,
        0.1F,
        0.1F,
        0.9F,
        0.8F,
        0.2F,
    });

    minifaiss::IndexIVFPQ ivfpq_index(2, 2, 1, 2);
    ivfpq_index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    ivfpq_index.add(std::array<float, 6>{
        0.9F,
        0.1F,
        0.1F,
        0.9F,
        0.8F,
        0.2F,
    });

    const std::array<float, 4> queries = {
        1.0F,
        0.0F,
        0.0F,
        1.0F,
    };

    const auto expected_flat = flat_index.search_batch(queries, 2);
    const auto expected_ivf = ivf_index.search_batch(queries, 2, 2);
    const auto expected_ivfpq = ivfpq_index.search_batch(queries, 2, 2);

    minifaiss::SearchExecutor executor(2, 4);
    auto flat_future =
        executor.submit([&] { return flat_index.search_batch(queries, 2); });
    auto ivf_future =
        executor.submit([&] { return ivf_index.search_batch(queries, 2, 2); });
    auto ivfpq_future = executor.submit(
        [&] { return ivfpq_index.search_batch(queries, 2, 2); });

    tests.check(same_batch_results(flat_future.get(), expected_flat),
                "async Flat batch search matches synchronous results");
    tests.check(same_batch_results(ivf_future.get(), expected_ivf),
                "async IVFFlat batch search matches synchronous results");
    tests.check(same_batch_results(ivfpq_future.get(), expected_ivfpq),
                "async IVFPQ batch search matches synchronous results");
}

}  // namespace

int main() {
    TestRunner tests;

    test_tasks_return_values_and_propagate_exceptions(tests);
    test_shutdown_drains_tasks_and_rejects_submissions(tests);
    test_parallel_batch_search_matches_scalar_results(tests);
    test_nested_executor_and_batch_parallelism(tests);
    test_async_batch_search_matches_synchronous_results(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All SearchExecutor tests passed.\n";
    }

    return tests.exit_code();
}
