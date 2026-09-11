#pragma once

#include "search_result.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace minifaiss {
class IndexIVFFlat {
public:
  IndexIVFFlat(std::size_t dimension, std::size_t nlist);

  [[nodiscard]] std::size_t dimension() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t nlist() const noexcept;
  [[nodiscard]] bool is_trained() const noexcept;

  void train(std::span<const float> training_vectors,
             std::size_t iterations = 10);

  void add(std::span<const float> vectors);

  [[nodiscard]] std::vector<SearchResult> search(std::span<const float> query,
                                                 std::size_t k,
                                                 std::size_t nprobe = 1) const;

private:
  struct InvertedList {
    std::vector<float> vectors;
    std::vector<std::size_t> ids;
  };

  std::size_t dimension_;
  std::size_t nlist_;
  std::size_t size_ = 0;
  bool trained_ = false;

  // nlist_ 条连续 row-major 中心向量。
  std::vector<float> centroids_;

  // 每个桶对应一个倒排链表。
  std::vector<InvertedList> lists_;
};

} // namespace minifaiss
