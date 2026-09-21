#include "minifaiss/index_ivf_pq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
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

    (void)index.add(std::array<float, 4>{0.2F, 0.8F, 10.2F, 10.8F});
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
        [&index] { (void)index.add(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}); },
        "add before training is rejected");
    tests.check(index.size() == 0, "rejected untrained add preserves size");
    tests.check_throws_logic_error(
        [&index] { (void)index.add(std::span<const float>{}); },
        "untrained empty add is rejected");

    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };
    index.train(training_vectors, 2);

    (void)index.add(std::array<float, 8>{
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

    (void)index.add(std::array<float, 4>{0.1F, 1.2F, 10.1F, 11.2F});
    tests.check(index.size() == 3, "later additions preserve global size");

    (void)index.add(std::span<const float>{});
    tests.check(index.size() == 3, "empty add is a no-op");

    tests.check_throws_invalid_argument(
        [&index] { (void)index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
        "incomplete add vector is rejected");
    tests.check(index.size() == 3, "malformed add preserves size");

    tests.check_throws_invalid_argument(
        [&index] {
            (void)index.add(std::array<float, 4>{
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

void test_index_ids_and_removal(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(2, 1, 1, 2);
    index.train(std::array<float, 4>{0.0F, 0.0F, 4.0F, 0.0F}, 1);

    const auto ids = index.add(std::array<float, 6>{
        0.0F, 0.0F, 2.0F, 0.0F, 4.0F, 0.0F,
    });
    tests.check(ids.size() == 3 && ids[0] == 0 && ids[1] == 1 &&
                    ids[2] == 2,
                "IVFPQ add returns consecutive index IDs");
    const auto moved_before = index.reconstruct(ids[2]);

    const std::array<minifaiss::IndexId, 3> ids_to_remove = {ids[1], 999,
                                                               ids[1]};
    tests.check(index.remove_ids(ids_to_remove) == 1,
                "IVFPQ removal ignores duplicate and unknown IDs");
    tests.check(index.size() == 2 && !index.contains(ids[1]),
                "IVFPQ removal invalidates only the removed ID");
    tests.check(index.contains(ids[2]),
                "IVFPQ swap-and-pop keeps the moved ID valid");
    tests.check_throws_out_of_range(
        [&index, &ids] { (void)index.reconstruct(ids[1]); },
        "IVFPQ cannot reconstruct a deleted ID");

    const auto moved_after = index.reconstruct(ids[2]);
    tests.check(moved_after == moved_before,
                "IVFPQ moves the complete PQ code block with its ID");
    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 3, 1);
    for (const auto& result : results) {
        tests.check(result.id != ids[1], "IVFPQ search excludes deleted IDs");
    }

    tests.check(index.remove_ids(std::array<minifaiss::IndexId, 1>{ids[2]}) ==
                    1,
                "IVFPQ removes a bucket-tail ID");
    tests.check(index.size() == 1 && !index.contains(ids[2]),
                "bucket-tail removal invalidates the ID and reduces size");
    tests.check_throws_out_of_range(
        [&index, &ids] { (void)index.reconstruct(ids[2]); },
        "IVFPQ cannot reconstruct a removed bucket-tail ID");

    tests.check_throws_invalid_argument(
        [&index] { (void)index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
        "invalid IVFPQ add does not consume an ID");
    const auto new_ids = index.add(std::array<float, 2>{1.0F, 0.0F});
    tests.check(new_ids.size() == 1 && new_ids[0] == 3,
                "IVFPQ does not reuse deleted IDs");
    tests.check(index.remove_ids(
                    std::span<const minifaiss::IndexId>{}) == 0,
                "empty IVFPQ removal is a no-op");
}

void test_reset_preserves_training_and_id_sequence(TestRunner& tests) {
    minifaiss::IndexIVFPQ index(2, 1, 1, 2);
    index.train(std::array<float, 4>{0.0F, 0.0F, 4.0F, 0.0F}, 1);
    const auto old_ids = index.add(std::array<float, 4>{1.0F, 0.0F, 3.0F, 0.0F});

    index.reset();
    tests.check(index.size() == 0, "IVFPQ reset clears indexed vectors");
    tests.check(index.is_trained(), "IVFPQ reset preserves training state");
    tests.check(!index.contains(old_ids[0]) && !index.contains(old_ids[1]),
                "IVFPQ reset invalidates old IDs");
    tests.check_throws_out_of_range(
        [&index, &old_ids] { (void)index.reconstruct(old_ids[0]); },
        "IVFPQ cannot reconstruct an ID invalidated by reset");

    const auto new_ids = index.add(std::array<float, 2>{2.0F, 0.0F});
    tests.check(new_ids.size() == 1 && new_ids[0] == 2,
                "IVFPQ reset does not reuse IDs");
    const auto reconstructed = index.reconstruct(new_ids[0]);
    tests.check(reconstructed.size() == 2,
                "IVFPQ reconstructs vectors added after reset");
    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 1, 1);
    tests.check(results.size() == 1 && results[0].id == new_ids[0],
                "IVFPQ accepts vectors without retraining after reset");

    minifaiss::IndexIVFPQ untrained_index(2, 1, 1, 2);
    untrained_index.reset();
    tests.check(!untrained_index.is_trained(),
                "untrained IVFPQ reset remains untrained");
    tests.check_throws_logic_error(
        [&untrained_index] {
            (void)untrained_index.add(std::array<float, 2>{1.0F, 0.0F});
        },
        "untrained IVFPQ reset still rejects add");
    tests.check_throws_logic_error(
        [&untrained_index] {
            (void)untrained_index.reconstruct(0);
        },
        "untrained IVFPQ reset still rejects reconstruct");
}

void test_save_and_load_preserve_index_state(TestRunner& tests) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "minifaiss_ivf_pq_v1_test.bin";
    std::filesystem::remove(path);

    minifaiss::IndexIVFPQ index(2, 1, 1, 2);
    index.train(std::array<float, 4>{0.0F, 0.0F, 4.0F, 0.0F}, 1);
    const auto ids = index.add(std::array<float, 6>{
        0.0F, 0.0F, 2.0F, 0.0F, 4.0F, 0.0F,
    });
    const std::vector<float> expected_reconstruction = index.reconstruct(ids[2]);
    const std::array<minifaiss::IndexId, 1> removed_ids = {ids[1]};
    (void)index.remove_ids(removed_ids);
    index.save(path);

    minifaiss::IndexIVFPQ loaded = minifaiss::IndexIVFPQ::load(path);
    tests.check(loaded.is_trained() && loaded.dimension() == 2 &&
                    loaded.nlist() == 1 && loaded.m() == 1 &&
                    loaded.ksub() == 2 && loaded.size() == 2,
                "IVFPQ load restores trained index state");
    tests.check(loaded.contains(ids[0]) && loaded.contains(ids[2]) &&
                    !loaded.contains(ids[1]),
                "IVFPQ load preserves ID holes");
    tests.check(loaded.reconstruct(ids[2]) == expected_reconstruction,
                "IVFPQ load restores PQ codes and codebooks");
    tests.check_throws_out_of_range(
        [&loaded, &ids] { (void)loaded.reconstruct(ids[1]); },
        "IVFPQ load keeps deleted IDs invalid");
    const auto new_ids = loaded.add(std::array<float, 2>{1.0F, 0.0F});
    tests.check(new_ids.size() == 1 && new_ids[0] == 3,
                "IVFPQ load preserves next ID");

    std::filesystem::remove(path);
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
    (void)index.add(std::array<float, 6>{
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

    (void)index.add(std::array<float, 6>{
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
    (void)index.add(std::array<float, 6>{
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
    (void)index.add(std::array<float, 8>{
        0.2F,
        0.8F,
        10.2F,
        10.8F,
        7.8F,
        9.2F,
        19.8F,
        21.2F,
    });
    (void)index.add(std::array<float, 4>{0.1F, 1.2F, 10.1F, 11.2F});

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
    test_index_ids_and_removal(tests);
    test_reset_preserves_training_and_id_sequence(tests);
    test_save_and_load_preserve_index_state(tests);
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
