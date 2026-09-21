#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
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
    [[nodiscard]] std::vector<IndexId> add(std::span<const float> vectors);
    [[nodiscard]] bool contains(IndexId id) const noexcept;
    [[nodiscard]] std::size_t remove_ids(std::span<const IndexId> ids);
    void reset();
    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static IndexIVFPQ load(const std::filesystem::path& path);

    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k,
        std::size_t nprobe = 1) const;
    [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
        std::span<const float> queries, std::size_t k,
        std::size_t nprobe = 1) const;

    [[nodiscard]] std::vector<float> reconstruct(IndexId id) const;

private:
    struct InvertedList {
        std::vector<IndexId> ids;
        std::vector<std::uint8_t> codes;
    };

    struct Location {
        std::size_t list_id;
        std::size_t local_id;
    };

    std::size_t dimension_;
    std::size_t nlist_;
    std::size_t size_ = 0;
    IndexId next_id_ = 0;
    bool trained_ = false;
    std::vector<float> centroids_;
    ProductQuantizer quantizer_;
    std::vector<InvertedList> lists_;
    std::vector<Location> locations_;
};

}  // namespace minifaiss
