#include "minifaiss/index_ivf_pq.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minifaiss {
namespace {

float inner_product(const float* lhs, const float* rhs, std::size_t dimension) {
    float score = 0.0F;
    for (std::size_t value_id = 0; value_id < dimension; ++value_id) {
        score += lhs[value_id] * rhs[value_id];
    }
    return score;
}

std::size_t select_best_centroid(const float* vector,
                                 const std::vector<float>& centroids,
                                 std::size_t dimension, std::size_t nlist) {
    std::size_t best_list_id = 0;
    float best_score = inner_product(vector, centroids.data(), dimension);

    for (std::size_t list_id = 1; list_id < nlist; ++list_id) {
        const float* centroid = centroids.data() + list_id * dimension;
        const float score = inner_product(vector, centroid, dimension);

        if (score > best_score) {
            best_score = score;
            best_list_id = list_id;
        }
    }

    return best_list_id;
}

struct CentroidScore {
    std::size_t list_id;
    float score;
};

bool is_better(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }
    return lhs.id < rhs.id;
}

struct WorseResultAtTop {
    bool operator()(const SearchResult& lhs, const SearchResult& rhs) const {
        return is_better(lhs, rhs);
    }
};

}  // namespace

IndexIVFPQ::IndexIVFPQ(std::size_t dimension, std::size_t nlist, std::size_t m,
                       std::size_t ksub)
    : dimension_(dimension),
      nlist_(nlist),
      quantizer_(dimension, m, ksub),
      lists_(nlist) {
    if (dimension_ == 0) {
        throw std::invalid_argument("dimension must be greater than zero");
    }
    if (nlist_ == 0) {
        throw std::invalid_argument("nlist must be greater than zero");
    }
}

std::size_t IndexIVFPQ::dimension() const noexcept { return dimension_; }

std::size_t IndexIVFPQ::size() const noexcept { return size_; }

std::size_t IndexIVFPQ::nlist() const noexcept { return nlist_; }

std::size_t IndexIVFPQ::m() const noexcept { return quantizer_.m(); }

std::size_t IndexIVFPQ::ksub() const noexcept { return quantizer_.ksub(); }

bool IndexIVFPQ::is_trained() const noexcept { return trained_; }

void IndexIVFPQ::train(std::span<const float> training_vectors,
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
    if (training_vector_count < quantizer_.ksub()) {
        throw std::invalid_argument(
            "training vector count must be at least ksub");
    }

    std::vector<float> new_centroids(nlist_ * dimension_);
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
            new_centroids[list_id * dimension_ + value_id] =
                training_vectors[list_id * dimension_ + value_id];
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

    std::vector<float> residuals(training_vectors.size());
    for (std::size_t vector_id = 0; vector_id < training_vector_count;
         ++vector_id) {
        const float* vector = training_vectors.data() + vector_id * dimension_;
        const std::size_t list_id =
            select_best_centroid(vector, new_centroids, dimension_, nlist_);
        const float* centroid = new_centroids.data() + list_id * dimension_;
        float* residual = residuals.data() + vector_id * dimension_;

        for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
            residual[value_id] = vector[value_id] - centroid[value_id];
        }
    }

    quantizer_.train(residuals, iterations);
    centroids_ = std::move(new_centroids);
    trained_ = true;
}

void IndexIVFPQ::add(std::span<const float> vectors) {
    if (!is_trained()) {
        throw std::logic_error("cannot add vectors to an untrained index");
    }
    if (vectors.empty()) {
        return;
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
    std::vector<std::size_t> list_ids(vector_count);
    std::vector<std::uint8_t> batch_codes(vector_count * m());
    std::vector<std::size_t> list_counts(nlist_, 0);
    std::vector<float> residual(dimension_);

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const float* vector = vectors.data() + vector_id * dimension_;
        const std::size_t list_id =
            select_best_centroid(vector, centroids_, dimension_, nlist_);
        const float* centroid = centroids_.data() + list_id * dimension_;

        for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
            residual[value_id] = vector[value_id] - centroid[value_id];
        }

        const std::vector<std::uint8_t> codes = quantizer_.encode(residual);
        list_ids[vector_id] = list_id;
        ++list_counts[list_id];

        for (std::size_t subquantizer_id = 0; subquantizer_id < m();
             ++subquantizer_id) {
            batch_codes[vector_id * m() + subquantizer_id] =
                codes[subquantizer_id];
        }
    }

    locations_.reserve(size_ + vector_count);
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        InvertedList& list = lists_[list_id];
        list.ids.reserve(list.ids.size() + list_counts[list_id]);
        list.codes.reserve(list.codes.size() + list_counts[list_id] * m());
    }

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const std::size_t list_id = list_ids[vector_id];
        InvertedList& list = lists_[list_id];
        const std::size_t local_id = list.ids.size();
        const std::size_t global_id = size_ + vector_id;

        list.ids.push_back(global_id);
        for (std::size_t subquantizer_id = 0; subquantizer_id < m();
             ++subquantizer_id) {
            list.codes.push_back(
                batch_codes[vector_id * m() + subquantizer_id]);
        }
        locations_.push_back({list_id, local_id});
    }

    size_ += vector_count;
}

std::vector<SearchResult> IndexIVFPQ::search(std::span<const float> query,
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
            {list_id, inner_product(query.data(), centroid, dimension_)});
    }

    std::sort(centroid_scores.begin(), centroid_scores.end(),
              [](const CentroidScore& lhs, const CentroidScore& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.list_id < rhs.list_id;
              });

    const std::vector<float> lut = quantizer_.inner_product_lut(query);
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        WorseResultAtTop>
        heap;

    for (std::size_t probe_id = 0; probe_id < nprobe; ++probe_id) {
        const CentroidScore centroid_score = centroid_scores[probe_id];
        const InvertedList& list = lists_[centroid_score.list_id];

        for (std::size_t local_id = 0; local_id < list.ids.size(); ++local_id) {
            float score = centroid_score.score;
            const std::size_t code_offset = local_id * m();

            for (std::size_t subquantizer_id = 0; subquantizer_id < m();
                 ++subquantizer_id) {
                const std::size_t code =
                    list.codes[code_offset + subquantizer_id];
                score += lut[subquantizer_id * ksub() + code];
            }

            const SearchResult candidate{list.ids[local_id], score};
            if (heap.size() < k) {
                heap.push(candidate);
            } else if (is_better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
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

std::vector<std::vector<SearchResult>> IndexIVFPQ::search_batch(
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
    std::vector<std::vector<SearchResult>> results;
    results.reserve(query_count);

    for (std::size_t query_id = 0; query_id < query_count; ++query_id) {
        const std::span<const float> query(
            queries.data() + query_id * dimension_, dimension_);
        results.push_back(search(query, k, nprobe));
    }

    return results;
}

std::vector<float> IndexIVFPQ::reconstruct(std::size_t id) const {
    if (!is_trained()) {
        throw std::logic_error("cannot reconstruct from an untrained index");
    }
    if (id >= size_) {
        throw std::out_of_range("vector ID is outside the index");
    }

    const Location location = locations_[id];
    const InvertedList& list = lists_[location.list_id];
    const std::size_t code_offset = location.local_id * m();
    const std::span<const std::uint8_t> codes(list.codes.data() + code_offset,
                                              m());

    std::vector<float> reconstructed = quantizer_.decode(codes);
    const float* centroid = centroids_.data() + location.list_id * dimension_;

    for (std::size_t value_id = 0; value_id < dimension_; ++value_id) {
        reconstructed[value_id] += centroid[value_id];
    }

    return reconstructed;
}

}  // namespace minifaiss
