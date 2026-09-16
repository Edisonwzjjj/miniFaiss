#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "search_result.hpp"

namespace minifaiss {

class IndexFlatIP {
public:
    explicit IndexFlatIP(std::size_t dimension);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    void add(std::span<const float> vectors);

    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                   std::size_t k) const;

private:
    std::size_t dimension_;
    std::vector<float> vectors_;
};

}  // namespace minifaiss
