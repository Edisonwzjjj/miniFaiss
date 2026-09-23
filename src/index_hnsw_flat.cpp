#include "minifaiss/index_hnsw_flat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "minifaiss/detail/dot_product.hpp"
#include "binary_io.hpp"
#include "parallel_for.hpp"

namespace minifaiss {
namespace {

constexpr std::size_t kInvalidLocation =
    std::numeric_limits<std::size_t>::max();
constexpr std::size_t kMaxLevel = 32;

bool is_better(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }
    return lhs.id < rhs.id;
}

std::uint64_t mix(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

}  // namespace

IndexHNSWFlat::IndexHNSWFlat(std::size_t dimension, std::size_t m,
                             std::size_t ef_construction, std::uint64_t seed)
    : dimension_(dimension),
      m_(m),
      ef_construction_(ef_construction),
      seed_(seed) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (m_ < 2) {
        throw std::invalid_argument("m must be at least two");
    }
    if (ef_construction_ == 0) {
        throw std::invalid_argument("ef_construction must be greater than zero");
    }
}

std::size_t IndexHNSWFlat::dimension() const noexcept { return dimension_; }
std::size_t IndexHNSWFlat::size() const noexcept { return size_; }
std::size_t IndexHNSWFlat::m() const noexcept { return m_; }
std::size_t IndexHNSWFlat::ef_construction() const noexcept { return ef_construction_; }
std::uint64_t IndexHNSWFlat::seed() const noexcept { return seed_; }

std::size_t IndexHNSWFlat::level_for(IndexId id) const noexcept {
    std::uint64_t value = mix(seed_ ^ static_cast<std::uint64_t>(id));
    std::size_t level = 0;
    while (level < kMaxLevel && value % m_ == 0U) {
        ++level;
        value = mix(value);
    }
    return level;
}

float IndexHNSWFlat::score(std::span<const float> query,
                           std::size_t node_position) const {
    return detail::dot_product(query.data(), nodes_[node_position].vector.data(),
                               dimension_);
}

std::size_t IndexHNSWFlat::greedy_search(std::span<const float> query,
                                         std::size_t entry,
                                         std::size_t level) const {
    std::size_t current = entry;
    bool improved = true;
    while (improved) {
        improved = false;
        const SearchResult current_result{nodes_[current].id, score(query, current)};
        for (const IndexId neighbor_id : nodes_[current].neighbors[level]) {
            const std::size_t neighbor = locations_[neighbor_id];
            const SearchResult candidate{nodes_[neighbor].id, score(query, neighbor)};
            if (is_better(candidate, current_result)) {
                current = neighbor;
                improved = true;
                break;
            }
        }
    }
    return current;
}

std::vector<std::size_t> IndexHNSWFlat::search_layer(
    std::span<const float> query, std::size_t entry, std::size_t level,
    std::size_t ef) const {
    std::vector<bool> visited(nodes_.size(), false);
    std::vector<std::size_t> frontier = {entry};
    std::vector<std::size_t> results;
    if (!nodes_[entry].deleted) {
        results.push_back(entry);
    }
    visited[entry] = true;

    while (!frontier.empty()) {
        auto best_it = std::max_element(
            frontier.begin(), frontier.end(), [this, &query](std::size_t lhs,
                                                             std::size_t rhs) {
                return is_better({nodes_[rhs].id, score(query, rhs)},
                                 {nodes_[lhs].id, score(query, lhs)});
            });
        const std::size_t current = *best_it;
        frontier.erase(best_it);

        if (!nodes_[current].deleted && results.size() >= ef) {
            const auto worst_it = std::min_element(
                results.begin(), results.end(), [this, &query](std::size_t lhs,
                                                               std::size_t rhs) {
                    return is_better({nodes_[lhs].id, score(query, lhs)},
                                     {nodes_[rhs].id, score(query, rhs)});
                });
            if (!is_better({nodes_[current].id, score(query, current)},
                           {nodes_[*worst_it].id, score(query, *worst_it)})) {
                continue;
            }
        }

        for (const IndexId neighbor_id : nodes_[current].neighbors[level]) {
            const std::size_t neighbor = locations_[neighbor_id];
            if (visited[neighbor]) {
                continue;
            }
            visited[neighbor] = true;
            frontier.push_back(neighbor);
            if (!nodes_[neighbor].deleted) {
                results.push_back(neighbor);
                if (results.size() > ef) {
                    const auto worst_it = std::min_element(
                        results.begin(), results.end(), [this, &query](std::size_t lhs,
                                                                       std::size_t rhs) {
                            return is_better({nodes_[lhs].id, score(query, lhs)},
                                             {nodes_[rhs].id, score(query, rhs)});
                        });
                    results.erase(worst_it);
                }
            }
        }
    }
    return results;
}

void IndexHNSWFlat::prune(std::size_t node_position, std::size_t level) {
    Node& node = nodes_[node_position];
    std::vector<IndexId>& neighbors = node.neighbors[level];
    std::sort(neighbors.begin(), neighbors.end(), [this, node_position](IndexId lhs,
                                                                        IndexId rhs) {
        const float lhs_score = detail::dot_product(
            nodes_[node_position].vector.data(), nodes_[locations_[lhs]].vector.data(),
            dimension_);
        const float rhs_score = detail::dot_product(
            nodes_[node_position].vector.data(), nodes_[locations_[rhs]].vector.data(),
            dimension_);
        return is_better({lhs, lhs_score}, {rhs, rhs_score});
    });
    while (neighbors.size() > m_) {
        const IndexId removed_id = neighbors.back();
        neighbors.pop_back();
        std::vector<IndexId>& reverse = nodes_[locations_[removed_id]].neighbors[level];
        reverse.erase(std::remove(reverse.begin(), reverse.end(), node.id), reverse.end());
    }
}

void IndexHNSWFlat::connect(std::size_t node_position,
                            std::size_t neighbor_position,
                            std::size_t level) {
    Node& node = nodes_[node_position];
    Node& neighbor = nodes_[neighbor_position];
    if (node.id == neighbor.id) {
        return;
    }
    if (std::find(node.neighbors[level].begin(), node.neighbors[level].end(),
                  neighbor.id) == node.neighbors[level].end()) {
        node.neighbors[level].push_back(neighbor.id);
    }
    if (std::find(neighbor.neighbors[level].begin(), neighbor.neighbors[level].end(),
                  node.id) == neighbor.neighbors[level].end()) {
        neighbor.neighbors[level].push_back(node.id);
    }
    prune(node_position, level);
    prune(neighbor_position, level);
}

std::vector<IndexId> IndexHNSWFlat::add(std::span<const float> vectors) {
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
    if (vector_count > std::numeric_limits<IndexId>::max() - next_id_) {
        throw std::overflow_error("index ID overflow");
    }

    std::vector<IndexId> assigned_ids;
    assigned_ids.reserve(vector_count);
    locations_.reserve(next_id_ + vector_count);
    nodes_.reserve(nodes_.size() + vector_count);
    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const IndexId id = next_id_++;
        const std::size_t level = level_for(id);
        const float* vector = vectors.data() + vector_id * dimension_;
        Node node{id, false, level,
                  std::vector<float>(vector, vector + dimension_),
                  std::vector<std::vector<IndexId>>(level + 1)};
        const std::size_t node_position = nodes_.size();
        nodes_.push_back(std::move(node));
        locations_.resize(next_id_, kInvalidLocation);
        locations_[id] = node_position;
        assigned_ids.push_back(id);

        if (size_ == 0) {
            entry_point_ = node_position;
            entry_level_ = level;
            ++size_;
            continue;
        }

        std::size_t current = entry_point_;
        for (std::size_t current_level = entry_level_; current_level > level;
             --current_level) {
            current = greedy_search(nodes_[node_position].vector, current,
                                    current_level);
        }
        const std::size_t highest_connection_level = std::min(level, entry_level_);
        for (std::size_t connection_level = highest_connection_level + 1;
             connection_level-- > 0;) {
            const std::vector<std::size_t> candidates = search_layer(
                nodes_[node_position].vector, current, connection_level,
                ef_construction_);
            std::vector<std::size_t> live_candidates;
            for (const std::size_t candidate : candidates) {
                if (!nodes_[candidate].deleted) {
                    live_candidates.push_back(candidate);
                }
            }
            std::sort(live_candidates.begin(), live_candidates.end(),
                      [this, node_position](std::size_t lhs, std::size_t rhs) {
                          const float lhs_score = detail::dot_product(
                              nodes_[node_position].vector.data(),
                              nodes_[lhs].vector.data(), dimension_);
                          const float rhs_score = detail::dot_product(
                              nodes_[node_position].vector.data(),
                              nodes_[rhs].vector.data(), dimension_);
                          return is_better({nodes_[lhs].id, lhs_score},
                                           {nodes_[rhs].id, rhs_score});
                      });
            if (live_candidates.size() > m_) {
                live_candidates.resize(m_);
            }
            for (const std::size_t neighbor : live_candidates) {
                connect(node_position, neighbor, connection_level);
            }
            if (!candidates.empty()) {
                current = candidates.front();
            }
        }
        if (level > entry_level_) {
            entry_point_ = node_position;
            entry_level_ = level;
        }
        ++size_;
    }
    return assigned_ids;
}

bool IndexHNSWFlat::contains(IndexId id) const noexcept {
    return id < locations_.size() && locations_[id] != kInvalidLocation &&
           !nodes_[locations_[id]].deleted;
}

std::size_t IndexHNSWFlat::remove_ids(std::span<const IndexId> ids) {
    std::size_t removed = 0;
    for (const IndexId id : ids) {
        if (!contains(id)) {
            continue;
        }
        nodes_[locations_[id]].deleted = true;
        --size_;
        ++removed;
    }
    return removed;
}

void IndexHNSWFlat::reset() {
    nodes_.clear();
    locations_.clear();
    size_ = 0;
    entry_point_ = kInvalidLocation;
    entry_level_ = 0;
}

void IndexHNSWFlat::save(const std::filesystem::path& path) const {
    detail::BinaryWriter payload;
    payload.write_u64(dimension_);
    payload.write_u64(m_);
    payload.write_u64(ef_construction_);
    payload.write_u64(seed_);
    payload.write_u64(size_);
    payload.write_u64(next_id_);
    payload.write_u8(entry_point_ == kInvalidLocation ? 0U : 1U);
    if (entry_point_ != kInvalidLocation) {
        payload.write_u64(nodes_[entry_point_].id);
        payload.write_u64(entry_level_);
    }
    payload.write_u64(nodes_.size());
    for (const Node& node : nodes_) {
        payload.write_u64(node.id);
        payload.write_u8(node.deleted ? 1U : 0U);
        payload.write_u64(node.level);
        for (const float value : node.vector) {
            payload.write_float(value);
        }
        for (const std::vector<IndexId>& neighbors : node.neighbors) {
            payload.write_u64(neighbors.size());
            for (const IndexId neighbor : neighbors) {
                payload.write_u64(neighbor);
            }
        }
    }
    detail::write_index_file(path, detail::IndexFileType::kHNSWFlat, payload);
}

IndexHNSWFlat IndexHNSWFlat::load(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes =
        detail::read_index_file(path, detail::IndexFileType::kHNSWFlat);
    detail::BinaryReader reader(bytes);
    const std::size_t dimension = detail::as_size(reader.read_u64());
    const std::size_t m = detail::as_size(reader.read_u64());
    const std::size_t ef_construction = detail::as_size(reader.read_u64());
    const std::uint64_t seed = reader.read_u64();
    const std::size_t size = detail::as_size(reader.read_u64());
    const IndexId next_id = detail::as_size(reader.read_u64());
    const std::uint8_t has_entry = reader.read_u8();
    if (dimension == 0 || m < 2 || ef_construction == 0 || has_entry > 1U ||
        next_id < size) {
        throw std::runtime_error("HNSW index file state is invalid");
    }
    IndexId entry_id = 0;
    std::size_t entry_level = 0;
    if (has_entry == 1U) {
        entry_id = detail::as_size(reader.read_u64());
        entry_level = detail::as_size(reader.read_u64());
    }
    const std::size_t node_count = detail::as_size(reader.read_u64());
    if ((node_count == 0) != (has_entry == 0U) || node_count < size) {
        throw std::runtime_error("HNSW index file node count is invalid");
    }

    IndexHNSWFlat index(dimension, m, ef_construction, seed);
    index.nodes_.reserve(node_count);
    index.locations_.resize(next_id, kInvalidLocation);
    std::size_t live_count = 0;
    for (std::size_t node_position = 0; node_position < node_count;
         ++node_position) {
        const IndexId id = detail::as_size(reader.read_u64());
        const std::uint8_t deleted_value = reader.read_u8();
        const std::size_t level = detail::as_size(reader.read_u64());
        if (id >= next_id || deleted_value > 1U || level > kMaxLevel ||
            index.locations_[id] != kInvalidLocation) {
            throw std::runtime_error("HNSW index file node is invalid");
        }
        Node node{id, deleted_value == 1U, level, std::vector<float>(dimension),
                  std::vector<std::vector<IndexId>>(level + 1)};
        for (float& value : node.vector) {
            value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("HNSW index file has non-finite vector");
            }
        }
        for (std::size_t layer = 0; layer <= level; ++layer) {
            const std::size_t neighbor_count = detail::as_size(reader.read_u64());
            if (neighbor_count > m) {
                throw std::runtime_error("HNSW index file neighbor count is invalid");
            }
            std::vector<IndexId>& neighbors = node.neighbors[layer];
            neighbors.resize(neighbor_count);
            for (IndexId& neighbor : neighbors) {
                neighbor = detail::as_size(reader.read_u64());
            }
        }
        index.locations_[id] = node_position;
        if (!node.deleted) {
            ++live_count;
        }
        index.nodes_.push_back(std::move(node));
    }
    if (live_count != size || !reader.empty()) {
        throw std::runtime_error("HNSW index file payload is invalid");
    }
    for (const Node& node : index.nodes_) {
        for (std::size_t layer = 0; layer <= node.level; ++layer) {
            for (const IndexId neighbor_id : node.neighbors[layer]) {
                if (neighbor_id >= next_id || neighbor_id == node.id ||
                    index.locations_[neighbor_id] == kInvalidLocation) {
                    throw std::runtime_error("HNSW index file neighbor is invalid");
                }
                const Node& neighbor = index.nodes_[index.locations_[neighbor_id]];
                if (neighbor.level < layer ||
                    std::find(neighbor.neighbors[layer].begin(),
                              neighbor.neighbors[layer].end(), node.id) ==
                        neighbor.neighbors[layer].end()) {
                    throw std::runtime_error("HNSW index file edge is not symmetric");
                }
            }
        }
    }
    if (has_entry == 1U) {
        if (entry_id >= next_id || index.locations_[entry_id] == kInvalidLocation) {
            throw std::runtime_error("HNSW index file entry point is invalid");
        }
        index.entry_point_ = index.locations_[entry_id];
        if (index.nodes_[index.entry_point_].level != entry_level) {
            throw std::runtime_error("HNSW index file entry level is invalid");
        }
        index.entry_level_ = entry_level;
    }
    index.size_ = size;
    index.next_id_ = next_id;
    return index;
}

std::vector<SearchResult> IndexHNSWFlat::search(std::span<const float> query,
                                                 std::size_t k,
                                                 std::size_t ef_search) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument("query dimension must match index dimension");
    }
    if (k == 0 || ef_search == 0 || ef_search < k) {
        throw std::invalid_argument("search parameters are invalid");
    }
    for (const float value : query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("query must contain only finite values");
        }
    }
    if (size_ == 0) {
        return {};
    }

    std::size_t current = entry_point_;
    for (std::size_t level = entry_level_; level > 0; --level) {
        current = greedy_search(query, current, level);
    }
    const std::vector<std::size_t> candidates = search_layer(query, current, 0,
                                                              ef_search);
    std::vector<SearchResult> results;
    for (const std::size_t candidate : candidates) {
        if (!nodes_[candidate].deleted) {
            results.push_back({nodes_[candidate].id, score(query, candidate)});
        }
    }
    std::sort(results.begin(), results.end(), is_better);
    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}

std::vector<std::vector<SearchResult>> IndexHNSWFlat::search_batch(
    std::span<const float> queries, std::size_t k, std::size_t ef_search) const {
    if (queries.size() % dimension_ != 0) {
        throw std::invalid_argument("query count must be divisible by dimension");
    }
    if (k == 0 || ef_search == 0 || ef_search < k) {
        throw std::invalid_argument("search parameters are invalid");
    }
    for (const float value : queries) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("queries must contain only finite values");
        }
    }
    const std::size_t query_count = queries.size() / dimension_;
    std::vector<std::vector<SearchResult>> results(query_count);
    detail::parallel_for(query_count, [this, queries, k, ef_search,
                                       &results](std::size_t query_id) {
        const std::span<const float> query(
            queries.data() + query_id * dimension_, dimension_);
        results[query_id] = search(query, k, ef_search);
    });
    return results;
}

}  // namespace minifaiss
