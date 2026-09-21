#include "minifaiss/index_flat_ip.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "minifaiss/detail/dot_product.hpp"
#include "binary_io.hpp"
#include "parallel_for.hpp"

namespace minifaiss {

namespace {

constexpr std::size_t kInvalidLocation =
    std::numeric_limits<std::size_t>::max();

bool is_better(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }

    return lhs.id < rhs.id;
}

struct WorseAtTop {
    bool operator()(const SearchResult& lhs, const SearchResult& rhs) const {
        return is_better(lhs, rhs);
    }
};

}  // namespace

IndexFlatIP::IndexFlatIP(std::size_t dimension) : dimension_(dimension) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
}

std::size_t IndexFlatIP::dimension() const noexcept { return dimension_; }

std::size_t IndexFlatIP::size() const noexcept { return size_; }

std::vector<IndexId> IndexFlatIP::add(std::span<const float> vectors) {
    if (vectors.empty()) {
        return {};
    }

    if (vectors.size() % dimension_ != 0) {
        throw std::invalid_argument(
            "vector count must be divisible by dimension");
    }
    for (const float value : vectors) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "vectors must contain only finite values");
        }
    }

    const std::size_t vector_count = vectors.size() / dimension_;
    if (vector_count > std::numeric_limits<IndexId>::max() - next_id_) {
        throw std::overflow_error("index ID overflow");
    }

    const IndexId first_id = next_id_;
    const IndexId next_id = first_id + vector_count;
    const std::size_t first_position = size_;

    vectors_.reserve(vectors_.size() + vectors.size());
    ids_.reserve(size_ + vector_count);
    locations_.reserve(next_id);

    std::vector<IndexId> assigned_ids;
    assigned_ids.reserve(vector_count);
    locations_.resize(next_id, kInvalidLocation);
    vectors_.insert(vectors_.end(), vectors.begin(), vectors.end());

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const IndexId id = first_id + vector_id;
        ids_.push_back(id);
        locations_[id] = first_position + vector_id;
        assigned_ids.push_back(id);
    }

    size_ += vector_count;
    next_id_ = next_id;
    return assigned_ids;
}

bool IndexFlatIP::contains(IndexId id) const noexcept {
    return id < locations_.size() && locations_[id] != kInvalidLocation;
}

std::size_t IndexFlatIP::remove_ids(std::span<const IndexId> ids) {
    std::size_t removed_count = 0;

    for (const IndexId id : ids) {
        if (!contains(id)) {
            continue;
        }

        const std::size_t position = locations_[id];
        const std::size_t last_position = size_ - 1;
        const IndexId moved_id = ids_[last_position];

        if (position != last_position) {
            float* destination =
                vectors_.data() + position * dimension_;
            const float* source =
                vectors_.data() + last_position * dimension_;
            std::copy_n(source, dimension_, destination);
            ids_[position] = moved_id;
            locations_[moved_id] = position;
        }

        vectors_.resize(vectors_.size() - dimension_);
        ids_.pop_back();
        locations_[id] = kInvalidLocation;
        --size_;
        ++removed_count;
    }

    return removed_count;
}

void IndexFlatIP::reset() {
    vectors_.clear();
    ids_.clear();
    locations_.clear();
    size_ = 0;
}

void IndexFlatIP::save(const std::filesystem::path& path) const {
    detail::BinaryWriter payload;
    payload.write_u64(dimension_);
    payload.write_u64(size_);
    payload.write_u64(next_id_);
    payload.write_u64(ids_.size());
    for (std::size_t position = 0; position < ids_.size(); ++position) {
        payload.write_u64(ids_[position]);
        const float* vector = vectors_.data() + position * dimension_;
        for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
            payload.write_float(vector[value_id]);
        }
    }
    detail::write_index_file(path, detail::IndexFileType::kFlatIP, payload);
}

IndexFlatIP IndexFlatIP::load(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes =
        detail::read_index_file(path, detail::IndexFileType::kFlatIP);
    detail::BinaryReader reader(bytes);
    const std::size_t dimension = detail::as_size(reader.read_u64());
    const std::size_t size = detail::as_size(reader.read_u64());
    const IndexId next_id = detail::as_size(reader.read_u64());
    const std::size_t entry_count = detail::as_size(reader.read_u64());
    if (dimension == 0 || entry_count != size || next_id < size) {
        throw std::runtime_error("index file state is invalid");
    }

    IndexFlatIP index(dimension);
    index.vectors_.reserve(detail::checked_product(size, dimension));
    index.ids_.reserve(size);
    index.locations_.resize(next_id, kInvalidLocation);
    for (std::size_t position = 0; position < entry_count; ++position) {
        const IndexId id = detail::as_size(reader.read_u64());
        if (id >= next_id || index.locations_[id] != kInvalidLocation) {
            throw std::runtime_error("index file ID is invalid");
        }
        for (std::size_t value_id = 0; value_id < dimension; ++value_id) {
            const float value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("index file contains non-finite vector");
            }
            index.vectors_.push_back(value);
        }
        index.ids_.push_back(id);
        index.locations_[id] = position;
    }
    if (!reader.empty()) {
        throw std::runtime_error("index file has trailing payload data");
    }
    index.size_ = size;
    index.next_id_ = next_id;
    return index;
}

std::vector<SearchResult> IndexFlatIP::search(std::span<const float> query,
                                              std::size_t k) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "query dimension must match index dimension");
    }
    for (const float value : query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "vectors must contain only finite values");
        }
    }
    if (k == 0) {
        throw std::invalid_argument("k must be greater than zero");
    }
    std::priority_queue<SearchResult, std::vector<SearchResult>, WorseAtTop>
        heap;

    for (std::size_t position = 0; position < size_; ++position) {
        const float* item = vectors_.data() + position * dimension_;
        const float score = detail::dot_product(query.data(), item, dimension_);
        const SearchResult candidate{ids_[position], score};

        if (heap.size() < k) {
            heap.push(candidate);
        } else if (is_better(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }

    std::vector<SearchResult> results;
    results.reserve(std::min(k, size_));
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }

    std::sort(results.begin(), results.end(), is_better);
    return results;
}

std::vector<std::vector<SearchResult>> IndexFlatIP::search_batch(
    std::span<const float> queries, std::size_t k) const {
    if (k == 0) {
        throw std::invalid_argument("k must be greater than zero");
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

    detail::parallel_for(
        query_count, [this, queries, k, &results](std::size_t query_id) {
            const std::span<const float> query(
                queries.data() + query_id * dimension_, dimension_);
            results[query_id] = search(query, k);
        });

    return results;
}

}  // namespace minifaiss
