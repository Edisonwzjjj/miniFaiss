#include <array>
#include <iostream>

#include "minifaiss/index_flat_ip.hpp"

int main() {
    minifaiss::IndexFlatIP index(2);

    index.add(std::array<float, 8>{
        1.0F,
        0.0F,
        0.0F,
        2.0F,
        3.0F,
        1.0F,
        3.0F,
        1.0F,
    });

    const std::array<float, 2> query = {1.0F, 1.0F};
    const auto results = index.search(query, 3);

    for (const auto& result : results) {
        std::cout << "id=" << result.id << ", score=" << result.score << '\n';
    }

    return 0;
}
