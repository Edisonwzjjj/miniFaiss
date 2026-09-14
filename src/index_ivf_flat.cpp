#include "minifaiss/index_ivf_flat.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minifaiss {
namespace {

float inner_product(const float* lhs, const float* rhs, std::size_t dimension) {
    float score = 0.0f;
    for (std::size_t d = 0; d < dimension; ++d) {
        score += lhs[d] * rhs[d];
    }
    return score;
}

std::size_t select_best_centroid(const float* vector,
                                 const std::vector<float>& centroids,
                                 std::size_t dimension,
                                 std::size_t nlist) {
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

void IndexIVFFlat::add(std::span<const float> vectors) {
    if (!is_trained()) {
        throw std::logic_error("not trained");
    }
    if (vectors.empty()) {
    return;
  }
    if (vectors.size() % dimension_ != 0) {
        throw std::invalid_argument("vector count must be divisible by dimension");
    }
    for (float vector : vectors) {
        if (!std::isfinite(vector)) {
            throw std::invalid_argument("vectors must contain only finite values");
        }
    }

    std::size_t vector_count = vectors.size() / dimension_;
    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const auto vector = vectors.data() + vector_id * dimension_;
        std::size_t centroid_id = select_best_centroid(vector, centroids_, dimension_, nlist_);
        auto& list = lists_[centroid_id];
        std::size_t global_id = size_ + vector_id;
        list.vectors.insert(list.vectors.end(), vector, vector + dimension_);
        list.ids.emplace_back(global_id);
    }

    size_ += vector_count;
}


std::vector<SearchResult> IndexIVFFlat::search(
    std::span<const float> query,
    std::size_t k,
    std::size_t nprobe) const {
  if (!is_trained()) {
    throw std::logic_error("cannot search an untrained index");
  }

  if (query.size() != dimension_) {
    throw std::invalid_argument("query dimension must match index dimension");
  }

  if (k == 0) {
    throw std::invalid_argument("k must be greater than zero");
  }

  if (nprobe == 0 || nprobe > nlist_) {
    throw std::invalid_argument("nprobe must be between one and nlist");
  }

  for (const float value : query) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("query must contain only finite values");
    }
  }

  std::vector<CentroidScore> centroid_scores;
  centroid_scores.reserve(nlist_);
  for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
    const float* centroid = centroids_.data() + list_id * dimension_;
    centroid_scores.push_back(
        {list_id, inner_product(query.data(), centroid, dimension_)});
  }

  std::sort(
      centroid_scores.begin(),
      centroid_scores.end(),
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
          inner_product(query.data(), item, dimension_),
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

  std::sort(
      results.begin(),
      results.end(),
      [](const SearchResult& lhs, const SearchResult& rhs) {
        if (lhs.score != rhs.score) {
          return lhs.score > rhs.score;
        }
        return lhs.id < rhs.id;
      });
  return results;
}

std::size_t IndexIVFFlat::dimension() const noexcept { return dimension_; }

std::size_t IndexIVFFlat::size() const noexcept { return size_; }

std::size_t IndexIVFFlat::nlist() const noexcept { return nlist_; }

bool IndexIVFFlat::is_trained() const noexcept { return trained_; }

}  // namespace minifaiss
