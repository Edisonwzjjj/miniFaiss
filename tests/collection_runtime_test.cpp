#include "minifaiss/collection_runtime.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
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

    [[nodiscard]] int exit_code() const { return failures_ == 0 ? 0 : 1; }

private:
    int failures_ = 0;
};

void test_constructor_and_sizes(TestRunner& tests) {
    const minifaiss::CollectionRuntime collection(2, 2, 4, 7);
    tests.check(collection.dimension() == 2, "collection stores dimension");
    tests.check(collection.main_size() == 0 && collection.delta_size() == 0,
                "new collection indexes are empty");
}

void test_durable_constructor(TestRunner& tests) {
    const std::filesystem::path wal_path =
        std::filesystem::temp_directory_path() / "minifaiss_collection_wal.bin";
    const minifaiss::CollectionRuntime collection(2, wal_path, 2, 4, 7);
    tests.check(collection.dimension() == 2 && collection.delta_size() == 0,
                "durable collection constructor initializes empty indexes");
}

void test_durable_add_requires_wal(TestRunner& tests) {
    minifaiss::CollectionRuntime collection(2, 2, 4, 0);
    tests.check_throws_logic_error(
        [&collection] {
            (void)collection.add_to_delta_durable(
                std::array<float, 2>{1.0F, 0.0F});
        },
        "durable add without WAL is rejected");
    tests.check_throws_logic_error(
        [&collection] { collection.recover_delta_from_wal(); },
        "recovery without WAL is rejected");
}

void test_durable_add_and_recover(TestRunner& tests) {
    const std::filesystem::path wal_path =
        std::filesystem::temp_directory_path() /
        "minifaiss_collection_recover_wal.bin";
    std::filesystem::remove(wal_path);

    std::vector<minifaiss::IndexId> first_ids;
    std::vector<minifaiss::IndexId> second_ids;
    {
        minifaiss::CollectionRuntime collection(2, wal_path, 2, 4, 0);
        first_ids = collection.add_to_delta_durable(
            std::array<float, 4>{1.0F, 0.0F, 3.0F, 0.0F});
        second_ids =
            collection.add_to_delta_durable(std::array<float, 2>{2.0F, 0.0F});
        tests.check(collection.add_to_delta_durable({}).empty(),
                    "empty durable add returns no IDs");
        tests.check_throws_invalid_argument(
            [&collection] {
                (void)collection.add_to_delta_durable(
                    std::array<float, 3>{1.0F, 0.0F, 2.0F});
            },
            "durable add rejects incomplete vectors");
        tests.check(collection.delta_size() == 3,
                    "durable add writes vectors to delta");
    }
    tests.check(first_ids.size() == 2 && first_ids[0] == 0 &&
                    first_ids[1] == 1 && second_ids.size() == 1 &&
                    second_ids[0] == 2,
                "durable add assigns consecutive delta IDs");

    minifaiss::CollectionRuntime restarted(2, wal_path, 2, 4, 0);
    restarted.recover_delta_from_wal();
    tests.check(restarted.delta_size() == 3,
                "recovery restores every durable delta vector");
    const auto results =
        restarted.search(std::array<float, 2>{1.0F, 0.0F}, 3, 4);
    tests.check(results.size() == 3 &&
                    results[0].source == minifaiss::CollectionSource::Delta &&
                    results[0].id == 1 && results[0].score == 3.0F &&
                    results[1].id == 2 && results[2].id == 0,
                "recovered delta vectors are searchable with original IDs");

    const auto next_ids =
        restarted.add_to_delta_durable(std::array<float, 2>{4.0F, 0.0F});
    tests.check(next_ids.size() == 1 && next_ids[0] == 3,
                "durable add after recovery continues ID sequence");
    tests.check_throws_logic_error(
        [&restarted] { restarted.recover_delta_from_wal(); },
        "recovery into a non-empty delta is rejected");

    minifaiss::CollectionRuntime restarted_again(2, wal_path, 2, 4, 0);
    restarted_again.recover_delta_from_wal();
    tests.check(restarted_again.delta_size() == 4,
                "recovery includes writes made after a previous recovery");

    std::filesystem::remove(wal_path);
}

void test_search_main_only(TestRunner& tests) {
    minifaiss::CollectionRuntime collection(2, 2, 4, 0);
    const auto ids = collection.add_to_main(
        std::array<float, 4>{1.0F, 0.0F, 3.0F, 0.0F});

    const auto results =
        collection.search(std::array<float, 2>{1.0F, 0.0F}, 2, 2);
    tests.check(results.size() == 2, "main-only search returns all main results");
    tests.check(results[0].source == minifaiss::CollectionSource::Main &&
                    results[0].id == ids[1] && results[0].score == 3.0F,
                "main-only result keeps source, ID, and score");
}

void test_search_delta_only(TestRunner& tests) {
    minifaiss::CollectionRuntime collection(2, 2, 4, 0);
    const auto ids = collection.add_to_delta(
        std::array<float, 4>{1.0F, 0.0F, 3.0F, 0.0F});

    const auto results =
        collection.search(std::array<float, 2>{1.0F, 0.0F}, 2, 2);
    tests.check(results.size() == 2, "delta-only search returns all delta results");
    tests.check(results[0].source == minifaiss::CollectionSource::Delta &&
                    results[0].id == ids[1] && results[0].score == 3.0F,
                "delta-only result keeps source, ID, and score");
}

void test_search_merges_global_top_k(TestRunner& tests) {
    minifaiss::CollectionRuntime collection(2, 2, 4, 0);
    const auto main_ids =
        collection.add_to_main(std::array<float, 4>{4.0F, 0.0F, 1.0F, 0.0F});
    const auto delta_ids = collection.add_to_delta(
        std::array<float, 4>{3.0F, 0.0F, 2.0F, 0.0F});

    const auto results =
        collection.search(std::array<float, 2>{1.0F, 0.0F}, 3, 4);
    tests.check(results.size() == 3, "merged search truncates to global k");
    tests.check(results[0].source == minifaiss::CollectionSource::Main &&
                    results[0].id == main_ids[0] && results[0].score == 4.0F,
                "global merge returns highest main score first");
    tests.check(results[1].source == minifaiss::CollectionSource::Delta &&
                    results[1].id == delta_ids[0] && results[1].score == 3.0F,
                "global merge orders delta result by score");
    tests.check(results[2].source == minifaiss::CollectionSource::Delta &&
                    results[2].id == delta_ids[1] && results[2].score == 2.0F,
                "global merge excludes lower result beyond k");

    const auto tie_results =
        collection.search(std::array<float, 2>{0.0F, 0.0F}, 2, 4);
    tests.check(tie_results[0].source == minifaiss::CollectionSource::Main &&
                    tie_results[0].id == 0,
                "equal scores use local ID then source as tie-breakers");
}

void test_search_validates_parameters(TestRunner& tests) {
    minifaiss::CollectionRuntime collection(2, 2, 4, 0);
    tests.check_throws_invalid_argument(
        [&collection] {
            (void)collection.search(std::array<float, 1>{1.0F}, 1, 1);
        },
        "collection rejects query with wrong dimension");
    tests.check_throws_invalid_argument(
        [&collection] {
            (void)collection.search(std::array<float, 2>{1.0F, 0.0F}, 0, 1);
        },
        "collection rejects zero k");
    tests.check_throws_invalid_argument(
        [&collection] {
            (void)collection.search(std::array<float, 2>{1.0F, 0.0F}, 2, 1);
        },
        "collection forwards invalid delta ef_search");
}

}  // namespace

int main() {
    TestRunner tests;
    test_constructor_and_sizes(tests);
    test_durable_constructor(tests);
    test_durable_add_requires_wal(tests);
    test_durable_add_and_recover(tests);
    test_search_main_only(tests);
    test_search_delta_only(tests);
    test_search_merges_global_top_k(tests);
    test_search_validates_parameters(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All CollectionRuntime tests passed.\n";
    }
    return tests.exit_code();
}
