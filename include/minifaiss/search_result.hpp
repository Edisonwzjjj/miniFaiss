#pragma once

#include <cstddef>

namespace minifaiss {

using IndexId = std::size_t;

struct SearchResult {
    IndexId id;
    float score;
};

}  // namespace minifaiss
