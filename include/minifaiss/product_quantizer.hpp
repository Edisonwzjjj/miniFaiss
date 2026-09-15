#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace minifaiss {

class ProductQuantizer {
public:
    ProductQuantizer(
        std::size_t dimension,
        std::size_t m,
        std::size_t ksub);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t m() const noexcept;
    [[nodiscard]] std::size_t ksub() const noexcept;
    [[nodiscard]] std::size_t subdimension() const noexcept;
    [[nodiscard]] bool has_codebooks() const noexcept;

    // Replaces all m * ksub codewords in row-major order.
    void set_codebooks(std::span<const float> codebooks);

    // Returns one codeword ID for every subvector.
    [[nodiscard]] std::vector<std::uint8_t> encode(
        std::span<const float> vector) const;

    // Reconstructs an approximate vector from m codeword IDs.
    [[nodiscard]] std::vector<float> decode(
        std::span<const std::uint8_t> codes) const;

    static float squared_l2_distance(
        const float* lhs,
        const float* rhs,
        std::size_t length);

private:
    std::size_t dimension_;
    std::size_t m_;
    std::size_t ksub_;
    std::size_t subdimension_;
    std::vector<float> codebooks_;
};

}  // namespace minifaiss
