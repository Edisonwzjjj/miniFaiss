#include "minifaiss/index_ivf_pq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

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

    template <typename Function>
    void check_throws_out_of_range(Function&& function, const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::out_of_range&) {
        }
    }

    [[nodiscard]] int exit_code() const { return failures_ == 0 ? 0 : 1; }

private:
    int failures_ = 0;
};

void test_constructor_and_initial_state(TestRunner& tests) {
    const minifaiss::IndexIVFPQ index(4, 2, 2, 2);

    tests.check(index.dimension() == 4, "constructor stores dimension");
    tests.check(index.nlist() == 2, "constructor stores nlist");
    tests.check(index.m() == 2, "constructor stores m");
    tests.check(index.ksub() == 2, "constructor stores ksub");
    tests.check(index.size() == 0, "new index has no vectors");
    tests.check(!index.is_trained(), "new index starts untrained");

    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFPQ index(0, 1, 1, 1); },
        "zero dimension is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFPQ index(4, 0, 1, 1); },
        "zero nlist is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFPQ index(4, 1, 0, 1); }, "zero m is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::IndexIVFPQ index(4, 1, 3, 1); },
        "dimension not divisible by m is rejected");
}

void test_train_learns_coarse_and_residual_models(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(4, 2, 2, 2);
    index.train(
        std::array<float, 16>{
            0.0F,
            0.0F,
            10.0F,
            10.0F,
            8.0F,
            8.0F,
            20.0F,
            20.0F,
            0.0F,
            2.0F,
            10.0F,
            12.0F,
            8.0F,
            10.0F,
            20.0F,
            22.0F,
        },
        2);

    tests.check(index.is_trained(),
                "successful training marks the index trained");
    tests.check(index.size() == 0, "training does not add indexed vectors");
}

void test_train_rejects_invalid_input(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(4, 2, 2, 2);

    tests.check_throws_invalid_argument([&index] { index.train({}, 1); },
                                        "empty training data is rejected");
    tests.check_throws_invalid_argument(
        [&index] { index.train(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1); },
        "incomplete training vector is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            index.train(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 1);
        },
        "fewer training vectors than nlist is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            index.train(
                std::array<float, 8>{
                    1.0F,
                    2.0F,
                    3.0F,
                    4.0F,
                    5.0F,
                    6.0F,
                    7.0F,
                    8.0F,
                },
                0);
        },
        "zero iterations are rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            index.train(
                std::array<float, 8>{
                    1.0F,
                    2.0F,
                    3.0F,
                    4.0F,
                    5.0F,
                    6.0F,
                    std::numeric_limits<float>::quiet_NaN(),
                    8.0F,
                },
                1);
        },
        "non-finite training data is rejected");
    tests.check(!index.is_trained(),
                "failed training preserves untrained state");
}

void test_reconstruct_validates_and_decodes_vectors(TestRunner& tests) {
    minifaiss::IndexIVFPQ untrained_index(4, 2, 2, 2);
    tests.check_throws_logic_error(
        [&untrained_index] { (void)untrained_index.reconstruct(0); },
        "reconstruct before training is rejected");

    minifaiss::IndexIVFPQ index(4, 2, 2, 2);
    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };
    index.train(training_vectors, 2);

    tests.check_throws_out_of_range(
        [&index] { (void)index.reconstruct(0); },
        "reconstruct rejects IDs outside an empty index");

    index.add(std::array<float, 4>{0.2F, 0.8F, 10.2F, 10.8F});
    const auto reconstructed = index.reconstruct(0);

    tests.check(reconstructed.size() == 4,
                "reconstruct returns the full vector dimension");
    for (const float value : reconstructed) {
        tests.check(std::isfinite(value), "reconstructed values are finite");
    }

    tests.check_throws_out_of_range(
        [&index] { (void)index.reconstruct(1); },
        "reconstruct rejects IDs beyond indexed vectors");
}

bool is_better(const minifaiss::SearchResult& lhs,
               const minifaiss::SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }
    return lhs.id < rhs.id;
}

float inner_product(std::span<const float> lhs, std::span<const float> rhs) {
    float score = 0.0F;
    for (std::size_t value_id = 0; value_id < lhs.size(); ++value_id) {
        score += lhs[value_id] * rhs[value_id];
    }
    return score;
}

std::vector<minifaiss::SearchResult> reconstruct_oracle(
    const minifaiss::IndexIVFPQ& index, std::span<const float> query,
    std::size_t k) {
    std::vector<minifaiss::SearchResult> results;
    results.reserve(index.size());

    for (std::size_t id = 0; id < index.size(); ++id) {
        const std::vector<float> reconstructed = index.reconstruct(id);
        results.push_back({id, inner_product(query, reconstructed)});
    }

    std::sort(results.begin(), results.end(), is_better);
    results.resize(std::min(k, results.size()));
    return results;
}

void test_add_encodes_vectors_and_preserves_size(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(4, 2, 2, 2);

    tests.check_throws_logic_error(
        [&index] { index.add(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}); },
        "add before training is rejected");
    tests.check(index.size() == 0, "rejected untrained add preserves size");

    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };
    index.train(training_vectors, 2);

    index.add(std::array<float, 8>{
        0.2F,
        0.8F,
        10.2F,
        10.8F,
        7.8F,
        9.2F,
        19.8F,
        21.2F,
    });
    tests.check(index.size() == 2, "adding two vectors increases size by two");

    index.add(std::array<float, 4>{0.1F, 1.2F, 10.1F, 11.2F});
    tests.check(index.size() == 3, "later additions preserve global size");

    index.add(std::span<const float>{});
    tests.check(index.size() == 3, "empty add is a no-op");

    tests.check_throws_invalid_argument(
        [&index] { index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
        "incomplete add vector is rejected");
    tests.check(index.size() == 3, "malformed add preserves size");

    tests.check_throws_invalid_argument(
        [&index] {
            index.add(std::array<float, 4>{
                1.0F,
                2.0F,
                std::numeric_limits<float>::infinity(),
                4.0F,
            });
        },
        "non-finite add vector is rejected");
    tests.check(index.size() == 3, "non-finite add preserves size");

    tests.check_throws_logic_error(
        [&index, &training_vectors] { index.train(training_vectors, 1); },
        "retraining after add is rejected");
}

void test_search_validates_input(TestRunner& tests) {
    minifaiss::IndexIVFPQ untrained_index(4, 2, 2, 2);
    tests.check_throws_logic_error(
        [&untrained_index] {
            (void)untrained_index.search(
                std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F}, 1);
        },
        "search before training is rejected");

    minifaiss::IndexIVFPQ index(4, 2, 2, 2);
    index.train(std::array<float, 8>{
        0.0F,
        0.0F,
        10.0F,
        10.0F,
        8.0F,
        8.0F,
        20.0F,
        20.0F,
    });

    const auto empty_results =
        index.search(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1, 1);
    tests.check(empty_results.empty(),
                "trained empty index returns no results");

    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1);
        },
        "short query is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(
                std::array<float, 4>{
                    1.0F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 4.0F},
                1);
        },
        "non-finite query is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 0);
        },
        "zero k is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 1,
                               0);
        },
        "zero nprobe is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 1,
                               3);
        },
        "nprobe larger than nlist is rejected");
}

void test_search_batch_matches_scalar_search(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(2, 2, 1, 2);
    index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    index.add(std::array<float, 6>{
        0.9F,
        0.1F,
        0.8F,
        0.2F,
        0.1F,
        0.9F,
    });

    const std::array<float, 4> queries = {1.0F, 0.0F, 0.0F, 1.0F};
    const auto batch_results = index.search_batch(queries, 10, 1);
    tests.check(batch_results.size() == 2,
                "IVFPQ batch search returns one row per query");

    for (std::size_t query_id = 0; query_id < batch_results.size();
         ++query_id) {
        const std::span<const float> query(queries.data() + query_id * 2, 2);
        const auto scalar_results = index.search(query, 10, 1);
        tests.check(batch_results[query_id].size() == scalar_results.size(),
                    "IVFPQ batch result count matches scalar search");
        for (std::size_t rank = 0; rank < scalar_results.size(); ++rank) {
            tests.check(
                batch_results[query_id][rank].id == scalar_results[rank].id,
                "IVFPQ batch result ID matches scalar search");
            tests.check(std::abs(batch_results[query_id][rank].score -
                                 scalar_results[rank].score) < 1e-5F,
                        "IVFPQ batch result score matches scalar search");
        }
    }

    tests.check(index.search_batch(std::span<const float>{}, 1, 1).empty(),
                "IVFPQ empty query batch returns no rows");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search_batch(std::array<float, 3>{1.0F, 0.0F, 1.0F}, 1,
                                     1);
        },
        "IVFPQ incomplete packed query batch is rejected");
    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.search_batch(
                std::array<float, 4>{
                    1.0F,
                    0.0F,
                    std::numeric_limits<float>::quiet_NaN(),
                    1.0F,
                },
                1, 1);
        },
        "IVFPQ non-finite later batch query is rejected");
    tests.check_throws_invalid_argument(
        [&index] { (void)index.search_batch(std::span<const float>{}, 1, 0); },
        "IVFPQ invalid nprobe rejects an empty query batch");

    minifaiss::IndexIVFPQ untrained_index(2, 2, 1, 2);
    tests.check_throws_logic_error(
        [&untrained_index] {
            (void)untrained_index.search_batch(std::span<const float>{}, 1, 1);
        },
        "IVFPQ untrained empty query batch is rejected");
}

void test_single_probe_searches_one_list(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(2, 2, 1, 2);
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
                "single probe returns only selected-list candidates");
    tests.check(results[0].id == 0,
                "single probe returns the best selected-list candidate first");
    tests.check(results[1].id == 1,
                "single probe excludes unprobed-list candidates");
}

void test_equal_scores_use_ascending_global_id(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(2, 1, 1, 1);
    index.train(std::array<float, 2>{1.0F, 0.0F}, 1);
    index.add(std::array<float, 6>{
        1.0F,
        0.0F,
        1.0F,
        0.0F,
        1.0F,
        0.0F,
    });

    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 2, 1);

    tests.check(results.size() == 2, "tie-break search returns k candidates");
    tests.check(results[0].id == 0,
                "tie-break keeps the smallest global ID first");
    tests.check(results[1].id == 1,
                "tie-break keeps the next-smallest global ID second");
    tests.check(results[0].score == results[1].score,
                "tie-break fixture produces equal approximate scores");
}

void test_full_probe_matches_reconstruction_oracle(TestRunner& tests) {
    constexpr std::size_t dimension = 4;
    constexpr std::size_t nlist = 2;
    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };

    minifaiss::IndexIVFPQ index(dimension, nlist, 2, 2);
    index.train(training_vectors, 2);
    index.add(std::array<float, 8>{
        0.2F,
        0.8F,
        10.2F,
        10.8F,
        7.8F,
        9.2F,
        19.8F,
        21.2F,
    });
    index.add(std::array<float, 4>{0.1F, 1.2F, 10.1F, 11.2F});

    const std::array<std::array<float, 4>, 2> queries = {{
        {1.0F, 0.0F, 0.0F, 1.0F},
        {0.0F, 1.0F, 1.0F, 0.0F},
    }};
    const std::array<std::size_t, 3> k_values = {1, 2, 10};

    for (const auto& query : queries) {
        for (const std::size_t k : k_values) {
            const auto expected = reconstruct_oracle(index, query, k);
            const auto actual = index.search(query, k, nlist);

            tests.check(
                actual.size() == expected.size(),
                "full-probe result count matches reconstruction oracle");
            if (actual.size() != expected.size()) {
                continue;
            }

            for (std::size_t result_id = 0; result_id < actual.size();
                 ++result_id) {
                tests.check(actual[result_id].id == expected[result_id].id,
                            "full-probe ID matches reconstruction oracle");
                tests.check(std::abs(actual[result_id].score -
                                     expected[result_id].score) < 1e-5F,
                            "full-probe score matches reconstruction oracle");
            }
        }
    }
}

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_initial_state(tests);
    test_train_learns_coarse_and_residual_models(tests);
    test_train_rejects_invalid_input(tests);
    test_reconstruct_validates_and_decodes_vectors(tests);
    test_add_encodes_vectors_and_preserves_size(tests);
    test_search_validates_input(tests);
    test_search_batch_matches_scalar_search(tests);
    test_single_probe_searches_one_list(tests);
    test_equal_scores_use_ascending_global_id(tests);
    test_full_probe_matches_reconstruction_oracle(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All IndexIVFPQ training tests passed.\n";
    }

    return tests.exit_code();
}
