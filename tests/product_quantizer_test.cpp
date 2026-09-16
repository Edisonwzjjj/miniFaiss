#include "minifaiss/product_quantizer.hpp"

#include <array>
#include <cstdint>
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

void set_example_codebooks(minifaiss::ProductQuantizer& quantizer) {
    quantizer.set_codebooks(std::array<float, 8>{
        0.0F,
        0.0F,
        1.0F,
        2.0F,
        10.0F,
        10.0F,
        20.0F,
        20.0F,
    });
}

void test_constructor_and_accessors(TestRunner& tests) {
    const minifaiss::ProductQuantizer quantizer(4, 2, 2);

    tests.check(quantizer.dimension() == 4, "constructor stores dimension");
    tests.check(quantizer.m() == 2, "constructor stores m");
    tests.check(quantizer.ksub() == 2, "constructor stores ksub");
    tests.check(quantizer.subdimension() == 2,
                "subdimension is dimension divided by m");
    tests.check(!quantizer.has_codebooks(), "new quantizer has no codebooks");

    tests.check_throws_invalid_argument(
        [] { minifaiss::ProductQuantizer quantizer(0, 1, 1); },
        "zero dimension is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::ProductQuantizer quantizer(4, 0, 1); },
        "zero m is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::ProductQuantizer quantizer(5, 2, 1); },
        "dimension not divisible by m is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::ProductQuantizer quantizer(4, 2, 0); },
        "zero ksub is rejected");
    tests.check_throws_invalid_argument(
        [] { minifaiss::ProductQuantizer quantizer(4, 2, 257); },
        "ksub larger than 256 is rejected");
}

void test_codebook_and_encode_decode(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(4, 2, 2);

    tests.check_throws_logic_error(
        [&quantizer] {
            (void)quantizer.encode(
                std::array<float, 4>{1.0F, 2.0F, 10.0F, 10.0F});
        },
        "encode without codebooks is rejected");
    tests.check_throws_logic_error(
        [&quantizer] {
            (void)quantizer.decode(std::array<std::uint8_t, 2>{0, 0});
        },
        "decode without codebooks is rejected");

    tests.check_throws_invalid_argument(
        [&quantizer] { quantizer.set_codebooks(std::array<float, 7>{}); },
        "wrong codebook size is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            quantizer.set_codebooks(std::array<float, 8>{
                0.0F,
                0.0F,
                1.0F,
                2.0F,
                10.0F,
                10.0F,
                std::numeric_limits<float>::quiet_NaN(),
                20.0F,
            });
        },
        "non-finite codebook value is rejected");

    set_example_codebooks(quantizer);
    tests.check(quantizer.has_codebooks(), "valid codebooks are stored");

    const auto codes =
        quantizer.encode(std::array<float, 4>{1.1F, 1.9F, 10.2F, 10.8F});
    tests.check(codes.size() == 2, "encode returns one code per subvector");
    tests.check(codes[0] == 1, "first subvector selects codeword one");
    tests.check(codes[1] == 0, "second subvector selects codeword zero");

    const auto decoded = quantizer.decode(codes);
    tests.check(decoded.size() == 4,
                "decode restores the full vector dimension");
    tests.check(decoded[0] == 1.0F && decoded[1] == 2.0F,
                "decode restores first codeword");
    tests.check(decoded[2] == 10.0F && decoded[3] == 10.0F,
                "decode restores second codeword");
}

void test_encode_decode_reject_invalid_input(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(4, 2, 2);
    set_example_codebooks(quantizer);

    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.encode(std::array<float, 3>{1.0F, 2.0F, 3.0F});
        },
        "short vector is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.encode(std::array<float, 4>{
                1.0F,
                2.0F,
                std::numeric_limits<float>::infinity(),
                4.0F,
            });
        },
        "non-finite vector is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.decode(std::array<std::uint8_t, 1>{0});
        },
        "wrong code count is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.decode(std::array<std::uint8_t, 2>{0, 2});
        },
        "code outside ksub is rejected");
}

void test_encode_ties_choose_lower_codeword(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(2, 1, 2);
    quantizer.set_codebooks(std::array<float, 4>{
        0.0F,
        0.0F,
        2.0F,
        0.0F,
    });

    const auto codes = quantizer.encode(std::array<float, 2>{1.0F, 0.0F});
    tests.check(codes[0] == 0, "equal distances choose the lower codeword ID");
}

void test_inner_product_lut(TestRunner& tests) {
    minifaiss::ProductQuantizer untrained_quantizer(4, 2, 2);
    tests.check_throws_logic_error(
        [&untrained_quantizer] {
            (void)untrained_quantizer.inner_product_lut(
                std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
        },
        "lookup table without codebooks is rejected");

    minifaiss::ProductQuantizer quantizer(4, 2, 2);
    set_example_codebooks(quantizer);

    const auto lut = quantizer.inner_product_lut(
        std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F});
    tests.check(lut.size() == 4, "lookup table has m times ksub entries");
    tests.check(lut[0] == 0.0F,
                "first subquantizer codeword zero score is correct");
    tests.check(lut[1] == 5.0F,
                "first subquantizer codeword one score is correct");
    tests.check(lut[2] == 70.0F,
                "second subquantizer codeword zero score is correct");
    tests.check(lut[3] == 140.0F,
                "second subquantizer codeword one score is correct");

    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.inner_product_lut(
                std::array<float, 3>{1.0F, 2.0F, 3.0F});
        },
        "short lookup query is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            (void)quantizer.inner_product_lut(std::array<float, 4>{
                1.0F,
                2.0F,
                std::numeric_limits<float>::quiet_NaN(),
                4.0F,
            });
        },
        "non-finite lookup query is rejected");
}

void test_train_learns_independent_codebooks(TestRunner& tests) {
    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };

    minifaiss::ProductQuantizer quantizer(4, 2, 2);
    quantizer.train(training_vectors, 2);

    tests.check(quantizer.has_codebooks(), "training creates codebooks");

    const auto low_codewords =
        quantizer.decode(std::array<std::uint8_t, 2>{0, 0});
    tests.check(low_codewords[0] == 0.0F && low_codewords[1] == 1.0F,
                "first subquantizer learns its low cluster mean");
    tests.check(low_codewords[2] == 10.0F && low_codewords[3] == 11.0F,
                "second subquantizer learns its low cluster mean");

    const auto high_codewords =
        quantizer.decode(std::array<std::uint8_t, 2>{1, 1});
    tests.check(high_codewords[0] == 8.0F && high_codewords[1] == 9.0F,
                "first subquantizer learns its high cluster mean");
    tests.check(high_codewords[2] == 20.0F && high_codewords[3] == 21.0F,
                "second subquantizer learns its high cluster mean");

    const auto codes =
        quantizer.encode(std::array<float, 4>{0.1F, 1.1F, 19.9F, 20.9F});
    tests.check(codes[0] == 0 && codes[1] == 1,
                "subquantizers independently select their nearest codewords");
}

void test_train_is_deterministic(TestRunner& tests) {
    const std::array<float, 16> training_vectors = {
        0.0F, 0.0F, 10.0F, 10.0F, 8.0F, 8.0F,  20.0F, 20.0F,
        0.0F, 2.0F, 10.0F, 12.0F, 8.0F, 10.0F, 20.0F, 22.0F,
    };

    minifaiss::ProductQuantizer first(4, 2, 2);
    minifaiss::ProductQuantizer second(4, 2, 2);
    first.train(training_vectors, 2);
    second.train(training_vectors, 2);

    const auto first_decoded = first.decode(std::array<std::uint8_t, 2>{1, 0});
    const auto second_decoded =
        second.decode(std::array<std::uint8_t, 2>{1, 0});
    tests.check(first_decoded == second_decoded,
                "identical training inputs produce identical codebooks");

    const auto first_codes =
        first.encode(std::array<float, 4>{7.9F, 9.1F, 10.1F, 10.9F});
    const auto second_codes =
        second.encode(std::array<float, 4>{7.9F, 9.1F, 10.1F, 10.9F});
    tests.check(first_codes == second_codes,
                "identical training inputs produce identical encodings");
}

void test_train_rejects_invalid_input_and_preserves_codebooks(
    TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(4, 2, 2);

    tests.check_throws_invalid_argument(
        [&quantizer] { quantizer.train({}, 1); },
        "empty training data is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            quantizer.train(std::array<float, 3>{1.0F, 2.0F, 3.0F}, 1);
        },
        "incomplete training vector is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            quantizer.train(std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}, 1);
        },
        "fewer training vectors than ksub is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            quantizer.train(
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
        "zero training iterations are rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] {
            quantizer.train(
                std::array<float, 8>{
                    1.0F,
                    2.0F,
                    3.0F,
                    4.0F,
                    5.0F,
                    6.0F,
                    std::numeric_limits<float>::infinity(),
                    8.0F,
                },
                1);
        },
        "non-finite training value is rejected");
    tests.check(!quantizer.has_codebooks(),
                "failed initial training does not create codebooks");

    set_example_codebooks(quantizer);
    tests.check_throws_invalid_argument(
        [&quantizer] { quantizer.train({}, 1); },
        "failed training rejects empty data after codebooks are set");

    const auto decoded = quantizer.decode(std::array<std::uint8_t, 2>{1, 0});
    tests.check(decoded[0] == 1.0F && decoded[1] == 2.0F &&
                    decoded[2] == 10.0F && decoded[3] == 10.0F,
                "failed training preserves existing codebooks");
}

void test_train_keeps_empty_codeword(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(1, 1, 2);
    quantizer.train(std::array<float, 2>{3.0F, 3.0F}, 2);

    const auto codeword_zero = quantizer.decode(std::array<std::uint8_t, 1>{0});
    const auto codeword_one = quantizer.decode(std::array<std::uint8_t, 1>{1});
    tests.check(codeword_zero[0] == 3.0F,
                "assigned codeword is updated from its cluster mean");
    tests.check(codeword_one[0] == 3.0F,
                "empty codeword retains its initialized value");
}

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_accessors(tests);
    test_codebook_and_encode_decode(tests);
    test_encode_decode_reject_invalid_input(tests);
    test_encode_ties_choose_lower_codeword(tests);
    test_inner_product_lut(tests);
    test_train_learns_independent_codebooks(tests);
    test_train_is_deterministic(tests);
    test_train_rejects_invalid_input_and_preserves_codebooks(tests);
    test_train_keeps_empty_codeword(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All ProductQuantizer tests passed.\n";
    }

    return tests.exit_code();
}
