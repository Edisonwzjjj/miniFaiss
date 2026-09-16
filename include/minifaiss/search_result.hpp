#pragma once

#include <cstddef>

namespace minifaiss {

struct SearchResult {
    std::size_t id;
    float score;
};

}  // namespace minifaiss
