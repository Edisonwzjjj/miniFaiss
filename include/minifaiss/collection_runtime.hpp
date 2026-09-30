#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "index_flat_ip.hpp"
#include "index_hnsw_flat.hpp"
#include "search_result.hpp"
#include "write_ahead_log.hpp"

namespace minifaiss {

enum class CollectionSource {
    Main,
    Delta,
};

struct CollectionSearchResult {
    CollectionSource source;
    IndexId id;
    float score;
};

class CollectionRuntime {
public:
    CollectionRuntime(std::size_t dimension, std::size_t hnsw_m = 16,
                      std::size_t hnsw_ef_construction = 64,
                      std::uint64_t hnsw_seed = 0);
    CollectionRuntime(std::size_t dimension, std::filesystem::path wal_path,
                      std::size_t hnsw_m = 16,
                      std::size_t hnsw_ef_construction = 64,
                      std::uint64_t hnsw_seed = 0);

    [[nodiscard]] std::size_t dimension() const noexcept;
    [[nodiscard]] std::size_t main_size() const noexcept;
    [[nodiscard]] std::size_t delta_size() const noexcept;

    [[nodiscard]] std::vector<IndexId> add_to_main(
        std::span<const float> vectors);
    [[nodiscard]] std::vector<IndexId> add_to_delta(
        std::span<const float> vectors);
    [[nodiscard]] std::vector<IndexId> add_to_delta_durable(
        std::span<const float> vectors);
    void recover_delta_from_wal();

    [[nodiscard]] std::vector<CollectionSearchResult> search(
        std::span<const float> query, std::size_t k,
        std::size_t delta_ef_search) const;

private:
    std::size_t dimension_{};
    IndexFlatIP main_index_;
    IndexHNSWFlat delta_index_;
    std::optional<WriteAheadLog> wal_;
    std::uint64_t next_sequence_number_ = 1;
    std::uint64_t next_operation_id_ = 1;
};

}  // namespace minifaiss
