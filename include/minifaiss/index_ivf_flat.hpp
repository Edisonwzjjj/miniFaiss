#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

#include "search_result.hpp"

namespace minifaiss {
class IndexIVFFlat {
public:
    IndexIVFFlat(std::size_t dimension, std::size_t nlist);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t nlist() const noexcept;
    [[nodiscard]] bool is_trained() const noexcept;

    void train(std::span<const float> training_vectors,
               std::size_t iterations = 10);

    [[nodiscard]] std::vector<IndexId> add(std::span<const float> vectors);
    [[nodiscard]] bool contains(IndexId id) const noexcept;
    [[nodiscard]] std::size_t remove_ids(std::span<const IndexId> ids);
    void reset();
    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static IndexIVFFlat load(const std::filesystem::path& path);

    [[nodiscard]] std::vector<SearchResult> search(
        std::span<const float> query, std::size_t k,
        std::size_t nprobe = 1) const;
    [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
        std::span<const float> queries, std::size_t k,
        std::size_t nprobe = 1) const;

private:
    struct InvertedList {
        std::vector<float> vectors;
        std::vector<IndexId> ids;
    };

    struct Location {
        std::size_t list_id;
        std::size_t local_id;
    };

    struct CentroidScore {
        std::size_t list_id;
        float score;
    };

    struct BetterSearchResult {
        bool operator()(const SearchResult& lhs,
                        const SearchResult& rhs) const {
            if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
            }
            return lhs.id < rhs.id;
        }
    };

    std::size_t dimension_;
    std::size_t nlist_;
    std::size_t size_ = 0;
    IndexId next_id_ = 0;
    bool trained_ = false;

    // nlist_ 条连续 row-major 中心向量。
    std::vector<float> centroids_;

    // 每个桶对应一个倒排链表。
    std::vector<InvertedList> lists_;
    std::vector<Location> locations_;
};

}  // namespace minifaiss
