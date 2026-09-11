#include "minifaiss/index_flat_ip.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

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
    void check_throws_invalid_argument(Function&& function, const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::invalid_argument&) {
        }
    }

    [[nodiscard]] int exit_code() const {
        return failures_ == 0 ? 0 : 1;
    }

private:
    int failures_ = 0;
};

void test_constructor(TestRunner& tests) {
    const minifaiss::IndexFlatIP index(3);
    tests.check(index.dimension() == 3, "constructor stores the dimension");
    tests.check(index.size() == 0, "new index is empty");

    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexFlatIP invalid_index(0); },
        "zero dimension is rejected");
}

void test_add_vectors(TestRunner& tests) {
    minifaiss::IndexFlatIP index(3);

    const std::array<float, 6> first_batch = {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };
    index.add(first_batch);
    tests.check(index.size() == 2, "adding two 3D vectors increases size by two");

    const std::array<float, 3> second_batch = {7.0F, 8.0F, 9.0F};
    index.add(second_batch);
    tests.check(index.size() == 3, "a later batch preserves previous vectors");

    index.add(std::span<const float>{});
    tests.check(index.size() == 3, "adding an empty batch is a no-op");
}

void test_add_rejects_invalid_input(TestRunner& tests) {
    minifaiss::IndexFlatIP index(3);
    index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F});

    tests.check_throws_invalid_argument(
        [&index] { index.add(std::array<float, 4>{4.0F, 5.0F, 6.0F, 7.0F}); },
        "a batch with incomplete vectors is rejected");
    tests.check(index.size() == 1, "malformed batch does not partially append");

    tests.check_throws_invalid_argument(
        [&index] {
            index.add(std::array<float, 3>{
                4.0F,
                std::numeric_limits<float>::quiet_NaN(),
                6.0F,
            });
        },
        "NaN values are rejected");
    tests.check(index.size() == 1, "non-finite batch does not partially append");

    tests.check_throws_invalid_argument(
        [&index] {
            index.add(std::array<float, 3>{
                4.0F,
                std::numeric_limits<float>::infinity(),
                6.0F,
            });
        },
        "infinite values are rejected");
    tests.check(index.size() == 1, "infinite batch does not partially append");
}

void test_search_by_inner_product(TestRunner& tests) {
    minifaiss::IndexFlatIP index(2);
    index.add(std::array<float, 8>{
        1.0F, 0.0F,
        0.0F, 2.0F,
        3.0F, 1.0F,
        3.0F, 1.0F,
    });

    const auto results = index.search(std::array<float, 2>{1.0F, 1.0F}, 3);

    tests.check(results.size() == 3, "search returns k results");
    tests.check(results[0].id == 2, "highest score ranks first");
    tests.check(results[0].score == 4.0F, "highest score is correct");
    tests.check(results[1].id == 3, "equal scores use ascending ID as tie-breaker");
    tests.check(results[1].score == 4.0F, "tied score is correct");
    tests.check(results[2].id == 1, "third-best result is correct");
    tests.check(results[2].score == 2.0F, "third-best score is correct");
}

void test_search_boundaries_and_invalid_input(TestRunner& tests) {
    minifaiss::IndexFlatIP empty_index(2);
    const auto empty_results = empty_index.search(std::array<float, 2>{1.0F, 2.0F}, 1);
    tests.check(empty_results.empty(), "empty index search returns no results");

    minifaiss::IndexFlatIP index(2);
    index.add(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F});

    const auto all_results = index.search(std::array<float, 2>{1.0F, 1.0F}, 5);
    tests.check(all_results.size() == 2, "k larger than index size returns all results");

    tests.check_throws_invalid_argument(
        [&index] { (void)index.search(std::array<float, 1>{1.0F}, 1); },
        "short query is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(
                std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 1);
        },
        "long query is rejected");
    tests.check_throws_invalid_argument(
        [&index] { (void)index.search(std::array<float, 2>{1.0F, 2.0F}, 0); },
        "zero k is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 2>{
                1.0F,
                std::numeric_limits<float>::quiet_NaN(),
            }, 1);
        },
        "non-finite query is rejected");
}

bool is_better(const minifaiss::SearchResult& lhs,
               const minifaiss::SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }

    return lhs.id < rhs.id;
}

std::vector<minifaiss::SearchResult> reference_search(
    std::span<const float> query,
    std::span<const float> items,
    std::size_t dimension,
    std::size_t k) {
    std::vector<minifaiss::SearchResult> results;
    const std::size_t item_count = items.size() / dimension;
    results.reserve(item_count);

    for (std::size_t id = 0; id < item_count; ++id) {
        float score = 0.0F;
        for (std::size_t value_id = 0; value_id < dimension; ++value_id) {
            score += query[value_id] * items[id * dimension + value_id];
        }
        results.push_back({id, score});
    }

    std::sort(results.begin(), results.end(), is_better);
    results.resize(std::min(k, results.size()));
    return results;
}

void test_search_matches_randomized_oracle(TestRunner& tests) {
    constexpr std::array<std::size_t, 4> dimensions = {1, 2, 7, 128};
    constexpr std::array<std::size_t, 4> item_counts = {1, 3, 17, 100};
    constexpr std::size_t query_count = 10;

    std::mt19937 generator(20260910);
    std::uniform_int_distribution<int> distribution(-3, 3);

    for (const std::size_t dimension : dimensions) {
        for (const std::size_t item_count : item_counts) {
            std::vector<float> items(item_count * dimension);
            for (float& value : items) {
                value = static_cast<float>(distribution(generator));
            }

            minifaiss::IndexFlatIP index(dimension);
            index.add(items);

            const std::array<std::size_t, 4> k_values = {
                1,
                2,
                item_count,
                item_count + 1,
            };

            for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
                std::vector<float> query(dimension);
                for (float& value : query) {
                    value = static_cast<float>(distribution(generator));
                }

                for (const std::size_t k : k_values) {
                    const auto actual = index.search(query, k);
                    const auto expected = reference_search(query, items, dimension, k);

                    tests.check(actual.size() == expected.size(),
                                "randomized search result count matches reference");
                    if (actual.size() != expected.size()) {
                        continue;
                    }

                    for (std::size_t result_id = 0; result_id < actual.size(); ++result_id) {
                        tests.check(actual[result_id].id == expected[result_id].id,
                                    "randomized search ID matches reference");
                        tests.check(actual[result_id].score == expected[result_id].score,
                                    "randomized search score matches reference");
                    }
                }
            }
        }
    }
}

}  // namespace

int main() {
    TestRunner tests;

    test_constructor(tests);
    test_add_vectors(tests);
    test_add_rejects_invalid_input(tests);
    test_search_by_inner_product(tests);
    test_search_boundaries_and_invalid_input(tests);
    test_search_matches_randomized_oracle(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All IndexFlatIP tests passed.\n";
    }

    return tests.exit_code();
}
