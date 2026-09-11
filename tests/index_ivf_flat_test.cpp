#include "minifaiss/index_ivf_flat.hpp"

#include <array>
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
    void check_throws_invalid_argument(Function&& function, const char* message) {
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

    [[nodiscard]] int exit_code() const {
        return failures_ == 0 ? 0 : 1;
    }

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
        [] { minifaiss::IndexIVFFlat index(1, 0); },
        "zero nlist is rejected");
}

void test_train_sets_trained_state(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);
    index.train(std::array<float, 8>{
        1.0F, 0.0F,
        0.0F, 1.0F,
        0.8F, 0.2F,
        0.1F, 0.9F,
    }, 2);

    tests.check(index.is_trained(), "valid training marks the index trained");
    tests.check(index.size() == 0, "training does not index training vectors");
}

void test_train_rejects_invalid_input(TestRunner& tests) {
    minifaiss::IndexIVFFlat index(2, 2);

    tests.check_throws_invalid_argument(
        [&index] { index.train({}, 1); },
        "empty training data is rejected");
    tests.check(!index.is_trained(), "failed initial training keeps index untrained");

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
            index.train(std::array<float, 4>{
                1.0F,
                0.0F,
                std::numeric_limits<float>::quiet_NaN(),
                1.0F,
            }, 1);
        },
        "non-finite training value is rejected");

    index.train(std::array<float, 4>{1.0F, 0.0F, 0.0F, 1.0F}, 1);
    tests.check(index.is_trained(), "valid training succeeds after rejected inputs");

    tests.check_throws_invalid_argument(
        [&index] { index.train(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1); },
        "failed retraining rejects malformed input");
    tests.check(index.is_trained(), "failed retraining preserves trained state");
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

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_initial_state(tests);
    test_constructor_rejects_invalid_arguments(tests);
    test_train_sets_trained_state(tests);
    test_train_rejects_invalid_input(tests);
    test_add_requires_training_and_preserves_size(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All IndexIVFFlat lifecycle tests passed.\n";
    }

    return tests.exit_code();
}
