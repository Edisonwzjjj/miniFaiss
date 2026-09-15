#include "minifaiss/product_quantizer.hpp"

#include <cmath>
#include <stdexcept>

namespace minifaiss {

ProductQuantizer::ProductQuantizer(
    std::size_t dimension,
    std::size_t m,
    std::size_t ksub)
    : dimension_(dimension),
      m_(m),
      ksub_(ksub),
      subdimension_(0) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (m_ == 0) {
        throw std::invalid_argument("m must be greater than zero");
    }
    if (dimension_ % m_ != 0) {
        throw std::invalid_argument("dimension must be divisible by m");
    }
    if (ksub_ == 0 || ksub_ > 256) {
        throw std::invalid_argument("ksub must be between one and 256");
    }

    subdimension_ = dimension_ / m_;
}

std::size_t ProductQuantizer::dimension() const noexcept {
    return dimension_;
}

std::size_t ProductQuantizer::m() const noexcept {
    return m_;
}

std::size_t ProductQuantizer::ksub() const noexcept {
    return ksub_;
}

std::size_t ProductQuantizer::subdimension() const noexcept {
    return subdimension_;
}

bool ProductQuantizer::has_codebooks() const noexcept {
    return !codebooks_.empty();
}

void ProductQuantizer::set_codebooks(std::span<const float> codebooks) {
    const std::size_t expected_size =
        m_ * ksub_ * subdimension_;

    if (codebooks.size() != expected_size) {
        throw std::invalid_argument("codebook size is invalid");
    }

    for (const float value : codebooks) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "codebooks must contain only finite values");
        }
    }

    codebooks_.assign(codebooks.begin(), codebooks.end());
}

std::vector<std::uint8_t> ProductQuantizer::encode(
    std::span<const float> vector) const {
    if (!has_codebooks()) {
        throw std::logic_error("codebooks must be set before encoding");
    }

    if (vector.size() != dimension_) {
        throw std::invalid_argument(
            "vector dimension must match ProductQuantizer dimension");
    }

    for (const float value : vector) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("vector must contain only finite values");
        }
    }

    std::vector<std::uint8_t> codes(m_, 0);
    for (std::size_t subquantizer_id = 0; subquantizer_id < m_;
         ++subquantizer_id) {
        const float* subvector =
            vector.data() + subquantizer_id * subdimension_;
        const float* first_codeword =
            codebooks_.data() + subquantizer_id * ksub_ * subdimension_;

        std::size_t best_code = 0;
        float best_distance =
            squared_l2_distance(subvector, first_codeword, subdimension_);

        for (std::size_t codeword_id = 1; codeword_id < ksub_;
             ++codeword_id) {
            const float* codeword =
                codebooks_.data() +
                (subquantizer_id * ksub_ + codeword_id) * subdimension_;
            const float distance =
                squared_l2_distance(subvector, codeword, subdimension_);

            if (distance < best_distance) {
                best_distance = distance;
                best_code = codeword_id;
            }
        }

        codes[subquantizer_id] = static_cast<std::uint8_t>(best_code);
    }

    return codes;
}

std::vector<float> ProductQuantizer::decode(
    std::span<const std::uint8_t> codes) const {
    if (!has_codebooks()) {
        throw std::logic_error("codebooks must be set before decoding");
    }

    if (codes.size() != m_) {
        throw std::invalid_argument("code count must match ProductQuantizer m");
    }

    std::vector<float> vector(dimension_);
    for (std::size_t subquantizer_id = 0; subquantizer_id < m_;
         ++subquantizer_id) {
        const std::size_t codeword_id = codes[subquantizer_id];
        if (codeword_id >= ksub_) {
            throw std::invalid_argument("codeword ID is outside the codebook");
        }

        float* subvector =
            vector.data() + subquantizer_id * subdimension_;
        const float* codeword =
            codebooks_.data() +
            (subquantizer_id * ksub_ + codeword_id) * subdimension_;

        for (std::size_t value_id = 0; value_id < subdimension_; ++value_id) {
            subvector[value_id] = codeword[value_id];
        }
    }

    return vector;
}

float ProductQuantizer::squared_l2_distance(
    const float* lhs,
    const float* rhs,
    std::size_t length) {
    float distance = 0.0F;
    for (std::size_t value_id = 0; value_id < length; ++value_id) {
        const float difference = lhs[value_id] - rhs[value_id];
        distance += difference * difference;
    }
    return distance;
}
}  // namespace minifaiss
