#include "minifaiss/index_ivf_pq.hpp"

#include <cmath>
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
    (void)query;
    (void)k;
    (void)nprobe;
    throw std::logic_error("IndexIVFPQ::search is not implemented");
}

std::vector<float> IndexIVFPQ::reconstruct(std::size_t id) const {
    (void)id;
    throw std::logic_error("IndexIVFPQ::reconstruct is not implemented");
}

}  // namespace minifaiss
