#include "minifaiss/collection_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace minifaiss {

CollectionRuntime::CollectionRuntime(std::size_t dimension, std::size_t hnsw_m,
                                     std::size_t hnsw_ef_construction,
                                     std::uint64_t hnsw_seed)
    : dimension_(dimension),
      main_index_(dimension),
      delta_index_(dimension, hnsw_m, hnsw_ef_construction, hnsw_seed) {}

CollectionRuntime::CollectionRuntime(std::size_t dimension,
                                     std::filesystem::path wal_path,
                                     std::size_t hnsw_m,
                                     std::size_t hnsw_ef_construction,
                                     std::uint64_t hnsw_seed)
    : dimension_(dimension),
      main_index_(dimension),
      delta_index_(dimension, hnsw_m, hnsw_ef_construction, hnsw_seed),
      wal_(std::in_place, std::move(wal_path)) {}

std::size_t CollectionRuntime::dimension() const noexcept {
    return dimension_;
}

std::size_t CollectionRuntime::main_size() const noexcept {
    return main_index_.size();
}

std::size_t CollectionRuntime::delta_size() const noexcept {
    return delta_index_.size();
}

std::vector<IndexId> CollectionRuntime::add_to_main(
    std::span<const float> vectors) {
    return main_index_.add(vectors);
}

std::vector<IndexId> CollectionRuntime::add_to_delta(
    std::span<const float> vectors) {
    return delta_index_.add(vectors);
}

std::vector<IndexId> CollectionRuntime::add_to_delta_durable(
    std::span<const float> vectors) {
    if (!wal_.has_value()) {
        throw std::logic_error("durable delta requires a WAL path");
    }
    if (vectors.empty()) {
        return {};
    }
    if (vectors.size() % dimension_ != 0) {
        throw std::invalid_argument("vector count must be divisible by dimension");
    }
    for (const float value : vectors) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("vectors must contain only finite values");
        }
    }
    const std::size_t vector_count = vectors.size() / dimension_;
    const IndexId first_id = delta_index_.next_id();
    if (vector_count >  std::numeric_limits<IndexId>::max() - first_id) {
        throw std::overflow_error("index ID overflow");
    }
    if (vector_count >
        std::numeric_limits<std::uint64_t>::max() - next_sequence_number_ + 1) {
        throw std::overflow_error("WAL sequence number overflow");
    }
    if (vector_count >
        std::numeric_limits<std::uint64_t>::max() - next_operation_id_ + 1) {
        throw std::overflow_error("WAL operation ID overflow");
    }

    std::vector<WalOperation> wal_operations;
    wal_operations.reserve(vector_count);
    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const IndexId current_id = first_id + vector_id;
        const float* begin = vectors.data() + vector_id * dimension_;
        wal_operations.push_back({next_sequence_number_ + vector_id,
                                  next_operation_id_ + vector_id,
                                  WalOperationType::Add, current_id,
                                  std::vector<float>(begin, begin + dimension_)});
    }

    std::vector<IndexId> assigned_ids;
    assigned_ids.reserve(vector_count);
    for (const WalOperation& operation : wal_operations) {
        wal_->append(operation);
        const std::vector<IndexId> ids = delta_index_.add(operation.vector);
        if (ids.size() != 1 || ids[0] != operation.id) {
            throw std::logic_error("WAL ID does not match Delta HNSW ID");
        }
        assigned_ids.push_back(ids[0]);
        ++next_sequence_number_;
        ++next_operation_id_;
    }
    return assigned_ids;
}

void CollectionRuntime::recover_delta_from_wal() {
    if (!wal_.has_value()) {
        throw std::logic_error("WAL recovery requires a WAL path");
    }
    if (delta_index_.next_id() != 0) {
        throw std::logic_error("WAL recovery requires an empty delta index");
    }

    const std::vector<WalOperation> replay_ops = wal_->replay();
    for (const WalOperation& operation : replay_ops) {
        if (operation.type == WalOperationType::Add) {
            if (operation.vector.size() != dimension_) {
                throw std::runtime_error("WAL vector dimension does not match");
            }
            const std::vector<IndexId> ids = delta_index_.add(operation.vector);
            if (ids.size() != 1 || ids[0] != operation.id) {
                throw std::runtime_error(
                    "WAL replay produced an unexpected index ID");
            }
        } else {
            const std::array<IndexId, 1> ids = {operation.id};
            (void)delta_index_.remove_ids(ids);
        }
        next_sequence_number_ =
            std::max(next_sequence_number_, operation.sequence_number + 1);
        next_operation_id_ =
            std::max(next_operation_id_, operation.operation_id + 1);
    }
}

std::vector<CollectionSearchResult> CollectionRuntime::search(
    std::span<const float> query, std::size_t k,
    std::size_t delta_ef_search) const {
    const std::vector<SearchResult> main_results = main_index_.search(query, k);
    const std::vector<SearchResult> delta_results =
        delta_index_.search(query, k, delta_ef_search);

    std::vector<CollectionSearchResult> combined;
    combined.reserve(main_results.size() + delta_results.size());
    for (const SearchResult& result : main_results) {
        combined.push_back({CollectionSource::Main, result.id, result.score});
    }
    for (const SearchResult& result : delta_results) {
        combined.push_back({CollectionSource::Delta, result.id, result.score});
    }

    std::sort(combined.begin(), combined.end(),
              [](const CollectionSearchResult& lhs,
                 const CollectionSearchResult& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  if (lhs.id != rhs.id) {
                      return lhs.id < rhs.id;
                  }
                  return lhs.source < rhs.source;
              });
    if (combined.size() > k) {
        combined.resize(k);
    }
    return combined;
}

}  // namespace minifaiss
