#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "search_result.hpp"

namespace minifaiss {

class IndexHNSWFlat {
public:
    IndexHNSWFlat(std::size_t dimension, std::size_t m = 16,
                  std::size_t ef_construction = 64, std::uint64_t seed = 0);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t m() const noexcept;
    [[nodiscard]] std::size_t ef_construction() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;

    [[nodiscard]] std::vector<IndexId> add(std::span<const float> vectors);
    [[nodiscard]] bool contains(IndexId id) const noexcept;
    [[nodiscard]] std::size_t remove_ids(std::span<const IndexId> ids);
    void reset();

    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static IndexHNSWFlat load(const std::filesystem::path& path);

    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k,
        std::size_t ef_search) const;
    [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
        std::span<const float> queries, std::size_t k,
        std::size_t ef_search) const;

    [[nodiscard]] IndexId next_id() const noexcept;

private:
    struct Node {
        IndexId id;
        bool deleted = false;
        std::size_t level;
        std::vector<float> vector;
        std::vector<std::vector<IndexId>> neighbors;
    };

    [[nodiscard]] std::size_t level_for(IndexId id) const noexcept;
    [[nodiscard]] float score(std::span<const float> query,
                              std::size_t node_position) const;
    [[nodiscard]] std::size_t greedy_search(std::span<const float> query,
                                            std::size_t entry,
                                            std::size_t level) const;
    [[nodiscard]] std::vector<std::size_t> search_layer(
        std::span<const float> query, std::size_t entry, std::size_t level,
        std::size_t ef) const;
    void connect(std::size_t node_position, std::size_t neighbor_position,
                 std::size_t level);
    void prune(std::size_t node_position, std::size_t level);

    std::size_t dimension_;
    std::size_t m_;
    std::size_t ef_construction_;
    std::uint64_t seed_;
    std::size_t size_ = 0;
    IndexId next_id_ = 0;
    std::vector<Node> nodes_;
    std::vector<std::size_t> locations_;
    std::size_t entry_point_ = static_cast<std::size_t>(-1);
    std::size_t entry_level_ = 0;
};

}  // namespace minifaiss
