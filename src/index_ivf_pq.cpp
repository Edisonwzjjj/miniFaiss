#include "minifaiss/index_ivf_pq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "binary_io.hpp"
#include "parallel_for.hpp"

namespace minifaiss {
namespace {

constexpr std::size_t kInvalidLocation =
    std::numeric_limits<std::size_t>::max();

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

std::vector<IndexId> IndexIVFPQ::add(std::span<const float> vectors) {
    if (!is_trained()) {
        throw std::logic_error("cannot add vectors to an untrained index");
    }
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

    const IndexId first_id = next_id_;
    const IndexId next_id = first_id + vector_count;
    locations_.reserve(next_id);
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        InvertedList& list = lists_[list_id];
        list.ids.reserve(list.ids.size() + list_counts[list_id]);
        list.codes.reserve(list.codes.size() + list_counts[list_id] * m());
    }

    std::vector<IndexId> assigned_ids;
    assigned_ids.reserve(vector_count);
    locations_.resize(next_id, {kInvalidLocation, kInvalidLocation});

    for (std::size_t vector_id = 0; vector_id < vector_count; ++vector_id) {
        const std::size_t list_id = list_ids[vector_id];
        InvertedList& list = lists_[list_id];
        const std::size_t local_id = list.ids.size();
        const IndexId id = first_id + vector_id;

        list.ids.push_back(id);
        for (std::size_t subquantizer_id = 0; subquantizer_id < m();
             ++subquantizer_id) {
            list.codes.push_back(
                batch_codes[vector_id * m() + subquantizer_id]);
        }
        locations_[id] = {list_id, local_id};
        assigned_ids.push_back(id);
    }

    size_ += vector_count;
    next_id_ = next_id;
    return assigned_ids;
}

bool IndexIVFPQ::contains(IndexId id) const noexcept {
    return id < locations_.size() &&
           locations_[id].list_id != kInvalidLocation;
}

std::size_t IndexIVFPQ::remove_ids(std::span<const IndexId> ids) {
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
            list.ids[location.local_id] = moved_id;
            std::uint8_t* destination =
                list.codes.data() + location.local_id * m();
            const std::uint8_t* source =
                list.codes.data() + last_local_id * m();
            std::copy_n(source, m(), destination);
            locations_[moved_id] = {location.list_id, location.local_id};
        }

        list.ids.pop_back();
        list.codes.resize(list.codes.size() - m());
        locations_[id] = {kInvalidLocation, kInvalidLocation};
        --size_;
        ++removed_count;

    }

    return removed_count;
}

void IndexIVFPQ::reset() {
    for (InvertedList& list : lists_) {
        list.ids.clear();
        list.codes.clear();
    }
    locations_.clear();
    size_ = 0;
}

void IndexIVFPQ::save(const std::filesystem::path& path) const {
    detail::BinaryWriter payload;
    payload.write_u64(dimension_);
    payload.write_u64(nlist_);
    payload.write_u64(m());
    payload.write_u64(ksub());
    payload.write_u64(size_);
    payload.write_u64(next_id_);
    payload.write_u8(trained_ ? 1U : 0U);
    if (trained_) {
        for (const float value : centroids_) {
            payload.write_float(value);
        }
        for (const float value : quantizer_.codebooks()) {
            payload.write_float(value);
        }
    }
    for (const InvertedList& list : lists_) {
        payload.write_u64(list.ids.size());
        for (std::size_t local_id = 0; local_id < list.ids.size(); ++local_id) {
            payload.write_u64(list.ids[local_id]);
            payload.write_bytes(list.codes.data() + local_id * m(), m());
        }
    }
    detail::write_index_file(path, detail::IndexFileType::kIVFPQ, payload);
}

IndexIVFPQ IndexIVFPQ::load(const std::filesystem::path& path) {
    const std::vector<std::uint8_t> bytes =
        detail::read_index_file(path, detail::IndexFileType::kIVFPQ);
    detail::BinaryReader reader(bytes);
    const std::size_t dimension = detail::as_size(reader.read_u64());
    const std::size_t nlist = detail::as_size(reader.read_u64());
    const std::size_t m = detail::as_size(reader.read_u64());
    const std::size_t ksub = detail::as_size(reader.read_u64());
    const std::size_t size = detail::as_size(reader.read_u64());
    const IndexId next_id = detail::as_size(reader.read_u64());
    const std::uint8_t trained_value = reader.read_u8();
    if (dimension == 0 || nlist == 0 || m == 0 || dimension % m != 0 ||
        ksub == 0 || ksub > 256 || trained_value > 1U || next_id < size) {
        throw std::runtime_error("index file state is invalid");
    }
    const bool trained = trained_value == 1U;
    if (!trained && size != 0) {
        throw std::runtime_error("untrained index file contains vectors");
    }

    IndexIVFPQ index(dimension, nlist, m, ksub);
    if (trained) {
        const std::size_t centroid_count = detail::checked_product(nlist, dimension);
        index.centroids_.resize(centroid_count);
        for (float& value : index.centroids_) {
            value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("index file contains non-finite centroid");
            }
        }
        const std::size_t codebook_count = detail::checked_product(
            detail::checked_product(m, ksub), dimension / m);
        std::vector<float> codebooks(codebook_count);
        for (float& value : codebooks) {
            value = reader.read_float();
            if (!std::isfinite(value)) {
                throw std::runtime_error("index file contains non-finite codebook");
            }
        }
        try {
            index.quantizer_.set_codebooks(codebooks);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("index file codebook is invalid");
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
        list.codes.reserve(detail::checked_product(list_size, m));
        for (std::size_t local_id = 0; local_id < list_size; ++local_id) {
            const IndexId id = detail::as_size(reader.read_u64());
            if (id >= next_id || index.locations_[id].list_id != kInvalidLocation) {
                throw std::runtime_error("index file ID is invalid");
            }
            std::vector<std::uint8_t> codes(m);
            reader.read_bytes(codes.data(), codes.size());
            for (const std::uint8_t code : codes) {
                if (code >= ksub) {
                    throw std::runtime_error("index file PQ code is invalid");
                }
            }
            list.ids.push_back(id);
            list.codes.insert(list.codes.end(), codes.begin(), codes.end());
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
    std::vector<std::vector<SearchResult>> results(query_count);

    detail::parallel_for(query_count, [this, queries, k, nprobe,
                                       &results](std::size_t query_id) {
        const std::span<const float> query(
            queries.data() + query_id * dimension_, dimension_);
        results[query_id] = search(query, k, nprobe);
    });

    return results;
}

std::vector<float> IndexIVFPQ::reconstruct(IndexId id) const {
    if (!is_trained()) {
        throw std::logic_error("cannot reconstruct from an untrained index");
    }
    if (!contains(id)) {
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
