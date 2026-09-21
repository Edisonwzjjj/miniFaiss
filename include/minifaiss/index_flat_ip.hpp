#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

#include "search_result.hpp"

namespace minifaiss {

class IndexFlatIP {
public:
    explicit IndexFlatIP(std::size_t dimension);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] std::vector<IndexId> add(std::span<const float> vectors);
    [[nodiscard]] bool contains(IndexId id) const noexcept;
    [[nodiscard]] std::size_t remove_ids(std::span<const IndexId> ids);
    void reset();
    void save(const std::filesystem::path& path) const;
    [[nodiscard]] static IndexFlatIP load(const std::filesystem::path& path);

    [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                   std::size_t k) const;
    [[nodiscard]] std::vector<std::vector<SearchResult>> search_batch(
        std::span<const float> queries, std::size_t k) const;

private:
    std::size_t dimension_;
    std::size_t size_ = 0;
    IndexId next_id_ = 0;
    std::vector<float> vectors_;
    std::vector<IndexId> ids_;
    std::vector<std::size_t> locations_;
};

}  // namespace minifaiss
