#pragma once

#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace minifaiss::detail {

[[nodiscard]] inline float dot_product_scalar(const float* lhs,
                                              const float* rhs,
                                              std::size_t length) noexcept {
    float sum = 0.0F;

    for (std::size_t value_id = 0; value_id < length; ++value_id) {
        sum += lhs[value_id] * rhs[value_id];
    }

    return sum;
}

[[nodiscard]] inline float dot_product(const float* lhs, const float* rhs,
                                       std::size_t length) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t accumulator = vdupq_n_f32(0.0F);
    std::size_t value_id = 0;

    for (; value_id + 4 <= length; value_id += 4) {
        accumulator = vfmaq_f32(accumulator, vld1q_f32(lhs + value_id),
                                vld1q_f32(rhs + value_id));
    }

    return vaddvq_f32(accumulator) + dot_product_scalar(lhs + value_id,
                                                        rhs + value_id,
                                                        length - value_id);
#else
    return dot_product_scalar(lhs, rhs, length);
#endif
}

}  // namespace minifaiss::detail
