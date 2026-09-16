#include "minifaiss/index_ivf_flat.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "minifaiss/index_flat_ip.hpp"

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
    void check_throws_invalid_argument(Function&& function,
                                       const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::invalid_argument&) {
        }
    }

    template <typename Function>
    void check_throws_logic_error(Function&& function, const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::logic_error&) {
        }
    }

    [[nodiscard]] int exit_code() const { return failures_ == 0 ? 0 : 1; }

private:
    int failures_ = 0;
};

void test_constructor_and_initial_state(TestRunner& tests) {
    const minifaiss::IndexIVFFlat index(3, 2);

    tests.check(index.dimension() == 3, "constructor stores dimension");
    tests.check(index.nlist() == 2, "constructor stores nlist");
    tests.check(index.size() == 0, "new index has no indexed vectors");
    tests.check(!index.is_trained(), "new index starts untrained");
}

void test_constructor_rejects_invalid_arguments(TestRunner& tests) {
    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFFlat index(0, 1); },
        "zero dimension is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFFlat index(1, 0); }, "zero nlist is rejected");
}

void test_train_sets_trained_state(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);
    index.train(
        std::array<float, 8>{
            1.0F,
            0.0F,
            0.0F,
            1.0F,
            0.8F,
            0.2F,
            0.1F,
            0.9F,
        },
        2);

    tests.check(index.is_trained(), "valid training marks the index trained");
    tests.check(index.size() == 0, "training does not index training vectors");
}

void test_train_rejects_invalid_input(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);

    tests.check_throws_invalid_argument([&index] { index.train({}, 1); },
                                        "empty training data is rejected");
    tests.check(!index.is_trained(),
                "failed initial training keeps index untrained");

    tests.check_throws_invalid_argument(
        [&index] { index.train(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1); },
        "incomplete training vector is rejected");
    tests.check_throws_invalid_argument(
        [&index] { index.train(std::array<float, 2>{1.0F, 2.0F}, 1); },
        "training data with fewer vectors than nlist is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 0);
        },
        "zero training iterations are rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            index.train(
                std::array<float, 4>{
                    1.0F,
                    0.0F,
                    std::numeric_limits<float>::quiet_NaN(),
                    1.0F,
                },
                1);
        },
        "non-finite training value is rejected");

    index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    tests.check(index.is_trained(),
                "valid training succeeds after rejected inputs");

    tests.check_throws_invalid_argument(
        [&index] { index.train(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1); },
        "failed retraining rejects malformed input");
    tests.check(index.is_trained(),
                "failed retraining preserves trained state");
    tests.check(index.size() == 0, "training validation never adds vectors");
}

void test_add_requires_training_and_preserves_size(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);

    tests.check_throws_logic_error(
        [&index] { index.add(std::array<float, 2>{1.0F, 0.0F}); },
        "add before training is rejected");
    tests.check(index.size() == 0, "rejected untrained add preserves size");

    index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    index.add(std::array<float, 4>{0.9F, 0.1F, 0.2F, 0.8F});
    tests.check(index.size() == 2, "adding two vectors increases size by two");

    index.add(std::array<float, 2>{0.8F, 0.2F});
    tests.check(index.size() == 3, "later additions retain global index size");

    index.add(std::span<const float>{});
    tests.check(index.size() == 3, "empty add is a no-op");

    tests.check_throws_invalid_argument(
        [&index] { index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
        "incomplete add vector is rejected");
    tests.check(index.size() == 3, "malformed add preserves size");

    tests.check_throws_invalid_argument(
        [&index] {
            index.add(std::array<float, 2>{
                1.0F,
                std::numeric_limits<float>::infinity(),
            });
        },
        "non-finite add vector is rejected");
    tests.check(index.size() == 3, "non-finite add preserves size");
}

void test_search_rejects_invalid_input(TestRunner& tests) {
    minifaiss::IndexIVFFlat untrained_index(2, 2);
    tests.check_throws_logic_error(
        [&untrained_index] {
            (void)untrained_index.search(std::array<float, 2>{1.0F, 0.0F}, 1);
        },
        "search before training is rejected");

    minifaiss::IndexIVFFlat index(2, 2);
    index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);

    const auto empty_results =
        index.search(std::array<float, 2>{1.0F, 0.0F}, 1, 1);
    tests.check(empty_results.empty(),
                "trained empty index returns no results");

    tests.check_throws_invalid_argument(
        [&index] { (void)index.search(std::array<float, 1>{1.0F}, 1); },
        "short query is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 4>{1.0F, 0.0F, 1.0F, 0.0F}, 1);
        },
        "long query is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(
                std::array<float, 2>{
                    1.0F,
                    std::numeric_limits<float>::quiet_NaN(),
                },
                1);
        },
        "non-finite query is rejected");
    tests.check_throws_invalid_argument(
        [&index] { (void)index.search(std::array<float, 2>{1.0F, 0.0F}, 0); },
        "zero k is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 2>{1.0F, 0.0F}, 1, 0);
        },
        "zero nprobe is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 2>{1.0F, 0.0F}, 1, 3);
        },
        "nprobe larger than nlist is rejected");
}

void test_single_probe_searches_one_list(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);
    index.train(
        std::array<float, 4>{
            1.0F,
            0.0F,
            0.0F,
            1.0F,
        },
        1);

    index.add(std::array<float, 6>{
        0.9F,
        0.1F,
        0.8F,
        0.2F,
        0.1F,
        0.9F,
    });

    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 10, 1);

    tests.check(results.size() == 2,
                "single probe returns only candidates in selected list");
    tests.check(results[0].id == 0,
                "single probe returns the best item in selected list");
    tests.check(results[0].score == 0.9F,
                "single probe returns the best selected-list score");
    tests.check(results[1].id == 1,
                "single probe excludes vectors in unprobed lists");
    tests.check(results[1].score == 0.8F,
                "single probe preserves selected-list ranking");
}

void test_full_probe_matches_flat_search(TestRunner& tests) {
    constexpr std::size_t dimension = 2;
    constexpr std::size_t nlist = 2;

    const std::array<float, 8> training_vectors = {
        1.0F, 0.0F, 0.0F, 1.0F, 0.8F, 0.2F, 0.1F, 0.9F,
    };
    const std::array<float, 12> item_vectors = {
        0.9F, 0.1F, 0.2F, 0.8F, 1.0F, 0.3F, 0.1F, 1.0F, -0.8F, 0.2F, 0.4F, 0.4F,
    };
    const std::array<std::array<float, 2>, 3> queries = {{
        {1.0F, 0.2F},
        {0.2F, 1.0F},
        {-1.0F, 0.1F},
    }};
    const std::array<std::size_t, 4> k_values = {1, 3, 6, 10};

    minifaiss::IndexFlatIP flat_index(dimension);
    flat_index.add(item_vectors);

    minifaiss::IndexIVFFlat ivf_index(dimension, nlist);
    ivf_index.train(training_vectors, 2);
    ivf_index.add(std::array<float, 6>{
        0.9F,
        0.1F,
        0.2F,
        0.8F,
        1.0F,
        0.3F,
    });
    ivf_index.add(std::array<float, 6>{
        0.1F,
        1.0F,
        -0.8F,
        0.2F,
        0.4F,
        0.4F,
    });

    for (const auto& query : queries) {
        for (const std::size_t k : k_values) {
            const auto expected = flat_index.search(query, k);
            const auto actual = ivf_index.search(query, k, nlist);

            tests.check(actual.size() == expected.size(),
                        "full-probe IVF result count matches Flat");
            if (actual.size() != expected.size()) {
                continue;
            }

            for (std::size_t result_id = 0; result_id < actual.size();
                 ++result_id) {
                tests.check(actual[result_id].id == expected[result_id].id,
                            "full-probe IVF ID matches Flat");
                tests.check(
                    actual[result_id].score == expected[result_id].score,
                    "full-probe IVF score matches Flat");
            }
        }
    }
}

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_initial_state(tests);
    test_constructor_rejects_invalid_arguments(tests);
    test_train_sets_trained_state(tests);
    test_train_rejects_invalid_input(tests);
    test_add_requires_training_and_preserves_size(tests);
    test_search_rejects_invalid_input(tests);
    test_single_probe_searches_one_list(tests);
    test_full_probe_matches_flat_search(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All IndexIVFFlat lifecycle tests passed.\n";
    }

    return tests.exit_code();
}
