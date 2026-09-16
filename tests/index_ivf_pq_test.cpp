#include "minifaiss/index_ivf_pq.hpp"

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

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_initial_state(tests);
    test_train_learns_coarse_and_residual_models(tests);
    test_train_rejects_invalid_input(tests);
    test_add_encodes_vectors_and_preserves_size(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All IndexIVFPQ training tests passed.\n";
    }

    return tests.exit_code();
}
