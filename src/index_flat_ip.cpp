#include "minifaiss/index_flat_ip.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <queue>
#include <stdexcept>
#include <vector>

#include "minifaiss/detail/dot_product.hpp"
#include "parallel_for.hpp"

namespace minifaiss {

IndexFlatIP::IndexFlatIP(std::size_t dimension) : dimension_(dimension) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
}

std::size_t IndexFlatIP::dimension() const noexcept { return dimension_; }

std::size_t IndexFlatIP::size() const noexcept {
    return vectors_.size() / dimension_;
}

void IndexFlatIP::add(std::span<const float> vectors) {
    if (vectors.empty()) {
        return;
    }

    if (vectors.size() % dimension_ != 0) {
        throw std::invalid_argument(
            "vector count must be divisible by dimension");
    }
    for (float value : vectors) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "vectors must contain only finite values");
        }
    }
    vectors_.insert(vectors_.end(), vectors.begin(), vectors.end());
}

namespace {

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

std::vector<SearchResult> IndexFlatIP::search(std::span<const float> query,
                                              std::size_t k) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument(
            "query dimension must match index dimension");
    }
    for (float value : query) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "vectors must contain only finite values");
        }
    }
    if (k == 0) {
        throw std::invalid_argument("k must be greater than zero");
    }
    const std::size_t total_ids = size();
    std::priority_queue<SearchResult, std::vector<SearchResult>, WorseAtTop>
        heap;

    for (std::size_t id = 0; id < total_ids; ++id) {
        const float* item = vectors_.data() + id * dimension_;
        const float score = detail::dot_product(query.data(), item, dimension_);
        const SearchResult candidate{id, score};

        if (heap.size() < k) {
            heap.push(candidate);
        } else if (is_better(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }

    std::vector<SearchResult> results;
    results.reserve(std::min(k, total_ids));
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
