#include "minifaiss/index_hnsw_flat.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
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

void test_constructor_and_add(TestRunner& tests) {
    minifaiss::IndexHNSWFlat index(2, 2, 4, 7);
    tests.check(index.dimension() == 2 && index.m() == 2 &&
                    index.ef_construction() == 4 && index.seed() == 7,
                "constructor stores HNSW configuration");

    const auto ids = index.add(std::array<float, 6>{
        1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F,
    });
    tests.check(ids.size() == 3 && ids[0] == 0 && ids[2] == 2,
                "HNSW add assigns consecutive IDs");
    tests.check(index.contains(ids[0]) && index.contains(ids[2]),
                "HNSW contains assigned IDs");
    tests.check_throws_invalid_argument(
        [&index] { (void)index.add(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
        "HNSW rejects incomplete vectors");
    const auto next_ids = index.add(std::array<float, 2>{4.0F, 0.0F});
    tests.check(next_ids.size() == 1 && next_ids[0] == 3,
                "HNSW failed add does not consume ID");
}

void test_search_and_batch(TestRunner& tests) {
    minifaiss::IndexHNSWFlat index(2, 4, 8, 0);
    const auto ids = index.add(std::array<float, 8>{
        1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F,
    });
    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 2, 8);
    tests.check(results.size() == 2 && results[0].id == ids[3] &&
                    results[1].id == ids[2],
                "HNSW search returns highest IP results");

    const std::array<float, 16> queries = {
        1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F,
        1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F,
    };
    const auto batch = index.search_batch(queries, 2, 8);
    tests.check(batch.size() == 8, "HNSW batch returns every query result");
    for (const auto& row : batch) {
        tests.check(row.size() == results.size() && row[0].id == results[0].id,
                    "HNSW batch matches scalar search");
    }
    tests.check_throws_invalid_argument(
        [&index] { (void)index.search(std::array<float, 2>{1.0F, 0.0F}, 2, 1); },
        "HNSW rejects ef_search below k");
}

void test_save_and_load(TestRunner& tests) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "minifaiss_hnsw_v1_test.bin";
    std::filesystem::remove(path);
    minifaiss::IndexHNSWFlat empty_index(2, 2, 8, 9);
    empty_index.save(path);
    minifaiss::IndexHNSWFlat empty_loaded = minifaiss::IndexHNSWFlat::load(path);
    tests.check(empty_loaded.size() == 0,
                "HNSW saves and loads an empty graph");
    std::filesystem::remove(path);

    minifaiss::IndexHNSWFlat index(2, 2, 8, 9);
    const auto ids = index.add(std::array<float, 6>{
        1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F,
    });
    const std::array<minifaiss::IndexId, 1> delete_ids = {ids[1]};
    (void)index.remove_ids(delete_ids);
    const auto before = index.search(std::array<float, 2>{1.0F, 0.0F}, 2, 8);
    index.save(path);

    minifaiss::IndexHNSWFlat loaded = minifaiss::IndexHNSWFlat::load(path);
    const auto after = loaded.search(std::array<float, 2>{1.0F, 0.0F}, 2, 8);
    tests.check(loaded.dimension() == 2 && loaded.m() == 2 &&
                    loaded.ef_construction() == 8 && loaded.seed() == 9 &&
                    loaded.size() == 2,
                "HNSW load restores configuration and size");
    tests.check(!loaded.contains(ids[1]) && loaded.contains(ids[2]),
                "HNSW load preserves logical deletion");
    tests.check(after.size() == before.size() && after[0].id == before[0].id &&
                    after[0].score == before[0].score,
                "HNSW load preserves search results");
    const auto new_ids = loaded.add(std::array<float, 2>{4.0F, 0.0F});
    tests.check(new_ids.size() == 1 && new_ids[0] == 3,
                "HNSW load preserves next ID");
    std::filesystem::remove(path);
}

void test_deleted_entry_remains_navigable(TestRunner& tests) {
    minifaiss::IndexHNSWFlat index(1, 2, 4, 2);
    const auto ids = index.add(std::array<float, 2>{1.0F, 2.0F});
    const std::array<minifaiss::IndexId, 1> deleted_ids = {ids[0]};
    (void)index.remove_ids(deleted_ids);
    const auto results = index.search(std::array<float, 1>{1.0F}, 1, 1);
    tests.check(results.size() == 1 && results[0].id == ids[1],
                "deleted entry remains available for HNSW navigation");
}

void test_lazy_deletion_and_reset(TestRunner& tests) {
    minifaiss::IndexHNSWFlat index(2, 2, 4, 1);
    const auto ids = index.add(std::array<float, 6>{
        1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F,
    });
    const std::array<minifaiss::IndexId, 3> delete_ids = {ids[2], 999, ids[2]};
    tests.check(index.remove_ids(delete_ids) == 1,
                "HNSW removal ignores unknown and duplicate IDs");
    tests.check(!index.contains(ids[2]) && index.size() == 2,
                "HNSW removal logically deletes node");
    const auto results = index.search(std::array<float, 2>{1.0F, 0.0F}, 2, 4);
    for (const auto& result : results) {
        tests.check(result.id != ids[2], "HNSW search excludes deleted node");
    }

    index.reset();
    tests.check(index.size() == 0 && !index.contains(ids[0]),
                "HNSW reset invalidates all IDs");
    const auto new_ids = index.add(std::array<float, 2>{4.0F, 0.0F});
    tests.check(new_ids.size() == 1 && new_ids[0] == 3,
                "HNSW reset does not reuse IDs");
}

}  // namespace

int main() {
    TestRunner tests;
    test_constructor_and_add(tests);
    test_search_and_batch(tests);
    test_save_and_load(tests);
    test_deleted_entry_remains_navigable(tests);
    test_lazy_deletion_and_reset(tests);
    if (tests.exit_code() == 0) {
        std::cout << "All IndexHNSWFlat tests passed.\n";
    }
    return tests.exit_code();
}
