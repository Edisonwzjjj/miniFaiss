#include "minifaiss/index_ivf_flat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "minifaiss/detail/dot_product.hpp"
#include "binary_io.hpp"
#include "parallel_for.hpp"

namespace minifaiss {
namespace {

constexpr std::size_t kInvalidLocation =
    std::numeric_limits<std::size_t>::max();

float inner_product(const float* lhs, const float* rhs, std::size_t dimension) {
    float score = 0.0F;
    for (std::size_t d = 0; d < dimension; ++d) {
        score += lhs[d] * rhs[d];
    }
    return score;
}

std::size_t select_best_centroid(const float* vector,
                                 const std::vector<float>& centroids,
                                 std::size_t dimension, std::size_t nlist) {
    std::size_t best_centroid = 0;
    float best_score = inner_product(vector, centroids.data(), dimension);
    for (std::size_t d = 1; d < nlist; ++d) {
        const float* centroid = centroids.data() + d * dimension;
        const float score = inner_product(vector, centroid, dimension);
        if (score > best_score) {
            best_score = score;
            best_centroid = d;
        }
    }
    return best_centroid;
}

}  // namespace

IndexIVFFlat::IndexIVFFlat(std::size_t dimension, std::size_t nlist)
    : dimension_(dimension), nlist_(nlist), lists_(nlist) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }

    if (nlist_ == 0) {
        throw std::invalid_argument("nlist must be greater than zero");
    }
}

void IndexIVFFlat::train(std::span<const float> training_vectors,
                         std::size_t iterations) {
    if (size_ != 0) {
        throw std::logic_error("cannot train an index that contains vectors");
    }

    if (iterations == 0) {
        throw std::invalid_argument("iterations must be greater than zero");
    }

    if (training_vectors.empty()) {
        throw std::invalid_argument("training vectors must not be empty");
    }

    if (training_vectors.size() % dimension_ != 0) {
        throw std::invalid_argument(
            "training vector count must be divisible by dimension");
    }

    for (const float value : training_vectors) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "training vectors must contain only finite values");
        }
    }

    const std::size_t training_vector_count =
        training_vectors.size() / dimension_;
    if (training_vector_count < nlist_) {
        throw std::invalid_argument(
            "training vector count must be at least nlist");
    }

    std::vector<float> new_centroids(nlist_ * dimension_);
    for (std::size_t centroid_id = 0; centroid_id < nlist_; ++centroid_id) {
        for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
            new_centroids[centroid_id * dimension_ + value_id] =
                training_vectors[centroid_id * dimension_ + value_id];
        }
    }

    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        std::vector<float> sums(nlist_ * dimension_, 0.0F);
        std::vector<std::size_t> counts(nlist_, 0);

        for (std::size_t vector_id = 0; vector_id < training_vector_count;
             ++vector_id) {
            const float* vector =
                training_vectors.data() + vector_id * dimension_;
            const std::size_t list_id =
                select_best_centroid(vector, new_centroids, dimension_, nlist_);

            ++counts[list_id];
            for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
                sums[list_id * dimension_ + value_id] += vector[value_id];
            }
        }

        for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
            if (counts[list_id] == 0) {
                continue;
            }

            const float count = static_cast<float>(counts[list_id]);
            for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
                new_centroids[list_id * dimension_ + value_id] =
                    sums[list_id * dimension_ + value_id] / count;
            }
        }
    }

    centroids_ = std::move(new_centroids);
    trained_ = true;
}

std::vector<IndexId> IndexIVFFlat::add(std::span<const float> vectors) {
    if (!is_trained()) {
        throw std::logic_error("not trained");
    }
    if (vectors.empty()) {
        return {};
    }
    if (vectors.size() % dimension_ != 0) {
        throw std::invalid_argument(
            "vector count must be divisible by dimension");
    }
    for (const float vector : vectors) {
        if (!std::isfinite(vector)) {
            throw std::invalid_argument(
                "vectors must contain only finite values");
        }
    }

    const std::size_t vector_count = vectors.size() / dimension_;
    if (vector_count > std::numeric_limits<IndexId>::max() - next_id_) {
        throw std::overflow_error("index ID overflow");
    }

    std::vector<std::size_t> list_ids(vector_count);
    std::vector<std::size_t> list_counts(nlist_, 0);
    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const float* vector = vectors.data() + vector_id * dimension_;
        const std::size_t list_id =
            select_best_centroid(vector, centroids_, dimension_, nlist_);
        list_ids[vector_id] = list_id;
        ++list_counts[list_id];
    }

    const IndexId first_id = next_id_;
    const IndexId next_id = first_id + vector_count;
    locations_.reserve(next_id);
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        InvertedList& list = lists_[list_id];
        list.vectors.reserve(list.vectors.size() +
                             list_counts[list_id] * dimension_);
        list.ids.reserve(list.ids.size() + list_counts[list_id]);
    }

    std::vector<IndexId> assigned_ids;
    assigned_ids.reserve(vector_count);
    locations_.resize(next_id, {kInvalidLocation, kInvalidLocation});

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const std::size_t list_id = list_ids[vector_id];
        InvertedList& list = lists_[list_id];
        const std::size_t local_id = list.ids.size();
        const IndexId id = first_id + vector_id;
        const float* vector = vectors.data() + vector_id * dimension_;

        list.vectors.insert(list.vectors.end(), vector, vector + dimension_);
        list.ids.push_back(id);
        locations_[id] = {list_id, local_id};
        assigned_ids.push_back(id);
    }

    size_ += vector_count;
    next_id_ = next_id;
    return assigned_ids;
}

bool IndexIVFFlat::contains(IndexId id) const noexcept {
    return id < locations_.size() &&
           locations_[id].list_id != kInvalidLocation;
}

std::size_t IndexIVFFlat::remove_ids(std::span<const IndexId> ids) {
    std::size_t removed_count = 0;

    for (const IndexId id : ids) {
        if (!contains(id)) {
            continue;
        }

        const Location location = locations_[id];
        InvertedList& list = lists_[location.list_id];
        const std::size_t last_local_id = list.ids.size() - 1;
        const IndexId moved_id = list.ids[last_local_id];

        if (location.local_id != last_local_id) {
            float* destination =
                list.vectors.data() + location.local_id * dimension_;
            const float* source =
                list.vectors.data() + last_local_id * dimension_;
            std::copy_n(source, dimension_, destination);
            list.ids[location.local_id] = moved_id;
            locations_[moved_id] = {location.list_id, location.local_id};
        }

        list.vectors.resize(list.vectors.size() - dimension_);
        list.ids.pop_back();
        locations_[id] = {kInvalidLocation, kInvalidLocation};
        --size_;
        ++removed_count;
    }

    return removed_count;
}

void IndexIVFFlat::reset() {
    for (InvertedList& list : lists_) {
        list.vectors.clear();
        list.ids.clear();
    }
    locations_.clear();
    size_ = 0;
}

void IndexIVFFlat::save(const std::filesystem::path& path) const {
    detail::BinaryWriter payload;
    payload.write_u64(dimension_);
    payload.write_u64(nlist_);
    payload.write_u64(size_);
    payload.write_u64(next_id_);
    payload.write_u8(trained_ ? 1U : 0U);
    if (trained_) {
        for (const float value : centroids_) {
            payload.write_float(value);
        }
    }
    for (const InvertedList& list : lists_) {
        payload.write_u64(list.ids.size());
        for (std::size_t local_id = 0; local_id < list.ids.size(); ++local_id) {
            payload.write_u64(list.ids[local_id]);
            const float* vector = list.vectors.data() + local_id * dimension_;
            for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
                payload.write_float(vector[value_id]);
            }
        }
    }
    detail::write_index_file(path, detail::IndexFileType::kIVFFlat, payload);
}

IndexIVFFlat IndexIVFFlat::load(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes =
        detail::read_index_file(path, detail::IndexFileType::kIVFFlat);
    detail::BinaryReader reader(bytes);
    const std::size_t dimension = detail::as_size(reader.read_u64());
    const std::size_t nlist = detail::as_size(reader.read_u64());
    const std::size_t size = detail::as_size(reader.read_u64());
    const IndexId next_id = detail::as_size(reader.read_u64());
    const std::uint8_t trained_value = reader.read_u8();
    if (dimension == 0 || nlist == 0 || trained_value > 1U || next_id < size) {
        throw std::runtime_error("index file state is invalid");
    }
    const bool trained = trained_value == 1U;
    if (!trained && size != 0) {
        throw std::runtime_error("untrained index file contains vectors");
    }

    IndexIVFFlat index(dimension, nlist);
    if (trained) {
        const std::size_t centroid_count = detail::checked_product(nlist, dimension);
        index.centroids_.resize(centroid_count);
        for (float& value : index.centroids_) {
            value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("index file contains non-finite centroid");
            }
        }
        index.trained_ = true;
    }
    index.locations_.resize(next_id, {kInvalidLocation, kInvalidLocation});
    std::size_t loaded_count = 0;
    for (std::size_t list_id = 0; list_id < nlist; ++list_id) {
        InvertedList& list = index.lists_[list_id];
        const std::size_t list_size = detail::as_size(reader.read_u64());
        if (list_size > size - loaded_count) {
            throw std::runtime_error("index file list size is invalid");
        }
        list.ids.reserve(list_size);
        list.vectors.reserve(detail::checked_product(list_size, dimension));
        for (std::size_t local_id = 0; local_id < list_size; ++local_id) {
            const IndexId id = detail::as_size(reader.read_u64());
            if (id >= next_id || index.locations_[id].list_id != kInvalidLocation) {
                throw std::runtime_error("index file ID is invalid");
            }
            for (std::size_t value_id = 0; value_id < dimension; ++value_id) {
                const float value = reader.read_float();
                if (!std::isfinite(value)) {
                    throw std::runtime_error("index file contains non-finite vector");
                }
                list.vectors.push_back(value);
            }
            list.ids.push_back(id);
            index.locations_[id] = {list_id, local_id};
        }
        loaded_count += list_size;
    }
    if (loaded_count != size || !reader.empty()) {
        throw std::runtime_error("index file payload is invalid");
    }
    index.size_ = size;
    index.next_id_ = next_id;
    return index;
}

std::vector<SearchResult> IndexIVFFlat::search(std::span<const float> query,
                                               std::size_t k,
                                               std::size_t nprobe) const {
    if (!is_trained()) {
        throw std::logic_error("cannot search an untrained index");
    }

    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "query dimension must match index dimension");
    }

    if (k == 0) {
        throw std::invalid_argument("k must be greater than zero");
    }

    if (nprobe == 0 || nprobe > nlist_) {
        throw std::invalid_argument("nprobe must be between one and nlist");
    }

    for (const float value : query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "query must contain only finite values");
        }
    }

    std::vector<CentroidScore> centroid_scores;
    centroid_scores.reserve(nlist_);
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        const float* centroid = centroids_.data() + list_id * dimension_;
        centroid_scores.push_back(
            {list_id, detail::dot_product(query.data(), centroid, dimension_)});
    }

    std::sort(centroid_scores.begin(), centroid_scores.end(),
              [](const CentroidScore& lhs, const CentroidScore& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.list_id < rhs.list_id;
              });

    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        BetterSearchResult>
        search_heap;

    for (std::size_t probe_id = 0; probe_id < nprobe; ++probe_id) {
        const InvertedList& list = lists_[centroid_scores[probe_id].list_id];
        const std::size_t vector_count = list.vectors.size() / dimension_;

        for (std::size_t local_id = 0; local_id < vector_count; ++local_id) {
            const float* item = list.vectors.data() + local_id * dimension_;
            const SearchResult candidate{
                list.ids[local_id],
                detail::dot_product(query.data(), item, dimension_),
            };

            if (search_heap.size() < k) {
                search_heap.push(candidate);
            } else if (BetterSearchResult{}(candidate, search_heap.top())) {
                search_heap.pop();
                search_heap.push(candidate);
            }
        }
    }

    std::vector<SearchResult> results;
    results.reserve(std::min(k, size_));
    while (!search_heap.empty()) {
        results.push_back(search_heap.top());
        search_heap.pop();
    }

    std::sort(results.begin(), results.end(),
              [](const SearchResult& lhs, const SearchResult& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.id < rhs.id;
              });
    return results;
}

std::vector<std::vector<SearchResult>> IndexIVFFlat::search_batch(
    std::span<const float> queries, std::size_t k, std::size_t nprobe) const {
    if (!is_trained()) {
        throw std::logic_error("cannot search an untrained index");
    }
    if (k == 0) {
        throw std::invalid_argument("k must be greater than zero");
    }
    if (nprobe == 0 || nprobe > nlist_) {
        throw std::invalid_argument("nprobe must be between one and nlist");
    }
    if (queries.size() % dimension_ != 0) {
        throw std::invalid_argument(
            "query count must be divisible by dimension");
    }
    for (const float value : queries) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "queries must contain only finite values");
        }
    }

    const std::size_t query_count = queries.size() / dimension_;
    std::vector<std::vector<SearchResult>> results(query_count);

    detail::parallel_for(query_count, [this, queries, k, nprobe,
                                       &results](std::size_t query_id) {
        const std::span<const float> query(
            queries.data() + query_id * dimension_, dimension_);
        results[query_id] = search(query, k, nprobe);
    });

    return results;
}

std::size_t IndexIVFFlat::dimension() const noexcept { return dimension_; }

std::size_t IndexIVFFlat::size() const noexcept { return size_; }

std::size_t IndexIVFFlat::nlist() const noexcept { return nlist_; }

bool IndexIVFFlat::is_trained() const noexcept { return trained_; }

}  // namespace minifaiss
