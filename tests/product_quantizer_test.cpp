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

void set_example_codebooks(minifaiss::ProductQuantizer& quantizer) {
    quantizer.set_codebooks(std::array<float, 8>{
        0.0F, 0.0F,
        1.0F, 2.0F,
        10.0F, 10.0F,
        20.0F, 20.0F,
    });
}

void test_constructor_and_accessors(TestRunner& tests) {
    const minifaiss::ProductQuantizer quantizer(4, 2, 2);

    tests.check(quantizer.dimension() == 4, "constructor stores dimension");
    tests.check(quantizer.m() == 2, "constructor stores m");
    tests.check(quantizer.ksub() == 2, "constructor stores ksub");
    tests.check(quantizer.subdimension() == 2, "subdimension is dimension divided by m");
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
        [&quantizer] { (void)quantizer.encode(std::array<float, 4>{1.0F, 2.0F, 10.0F, 10.0F}); },
        "encode without codebooks is rejected");
    tests.check_throws_logic_error(
        [&quantizer] { (void)quantizer.decode(std::array<std::uint8_t, 2>{0, 0}); },
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

    const auto codes = quantizer.encode(
        std::array<float, 4>{1.1F, 1.9F, 10.2F, 10.8F});
    tests.check(codes.size() == 2, "encode returns one code per subvector");
    tests.check(codes[0] == 1, "first subvector selects codeword one");
    tests.check(codes[1] == 0, "second subvector selects codeword zero");

    const auto decoded = quantizer.decode(codes);
    tests.check(decoded.size() == 4, "decode restores the full vector dimension");
    tests.check(decoded[0] == 1.0F && decoded[1] == 2.0F,
                "decode restores first codeword");
    tests.check(decoded[2] == 10.0F && decoded[3] == 10.0F,
                "decode restores second codeword");
}

void test_encode_decode_reject_invalid_input(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(4, 2, 2);
    set_example_codebooks(quantizer);

    tests.check_throws_invalid_argument(
        [&quantizer] { (void)quantizer.encode(std::array<float, 3>{1.0F, 2.0F, 3.0F}); },
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
        [&quantizer] { (void)quantizer.decode(std::array<std::uint8_t, 1>{0}); },
        "wrong code count is rejected");
    tests.check_throws_invalid_argument(
        [&quantizer] { (void)quantizer.decode(std::array<std::uint8_t, 2>{0, 2}); },
        "code outside ksub is rejected");
}

void test_encode_ties_choose_lower_codeword(TestRunner& tests) {
    minifaiss::ProductQuantizer quantizer(2, 1, 2);
    quantizer.set_codebooks(std::array<float, 4>{
        0.0F, 0.0F,
        2.0F, 0.0F,
    });

    const auto codes = quantizer.encode(std::array<float, 2>{1.0F, 0.0F});
    tests.check(codes[0] == 0, "equal distances choose the lower codeword ID");
}

}  // namespace

int main() {
    TestRunner tests;

    test_constructor_and_accessors(tests);
    test_codebook_and_encode_decode(tests);
    test_encode_decode_reject_invalid_input(tests);
    test_encode_ties_choose_lower_codeword(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All ProductQuantizer tests passed.\n";
    }

    return tests.exit_code();
}
