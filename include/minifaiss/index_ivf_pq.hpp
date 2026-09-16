#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "product_quantizer.hpp"
#include "search_result.hpp"

namespace minifaiss {

class IndexIVFPQ {
public:
    IndexIVFPQ(std::size_t dimension, std::size_t nlist, std::size_t m,
               std::size_t ksub);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t nlist() const noexcept;
    [[nodiscard]] std::size_t m() const noexcept;
    [[nodiscard]] std::size_t ksub() const noexcept;
    [[nodiscard]] bool is_trained() const noexcept;

    void train(std::span<const float> training_vectors,
               std::size_t iterations = 10);
    void add(std::span<const float> vectors);

    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k,
        std::size_t nprobe = 1) const;

    [[nodiscard]] std::vector<float> reconstruct(std::size_t id) const;

private:
    struct InvertedList {
        std::vector<std::size_t> ids;
        std::vector<std::uint8_t> codes;
    };

    struct Location {
        std::size_t list_id;
        std::size_t local_id;
    };

    std::size_t dimension_;
    std::size_t nlist_;
    std::size_t size_ = 0;
    bool trained_ = false;
    std::vector<float> centroids_;
    ProductQuantizer quantizer_;
    std::vector<InvertedList> lists_;
    std::vector<Location> locations_;
};

}  // namespace minifaiss
