#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace minifaiss::detail {

template <typename Function>
void parallel_for(std::size_t count, Function&& function) {
    constexpr std::size_t kMinimumParallelQueries = 8;
    constexpr std::size_t kMaximumWorkers = 8;

    if (count < kMinimumParallelQueries) {
        for (std::size_t index = 0; index < count; ++index) {
            function(index);
        }
        return;
    }

    const unsigned int hardware_count = std::thread::hardware_concurrency();
    const std::size_t available_workers =
        hardware_count == 0 ? 1 : static_cast<std::size_t>(hardware_count);
    const std::size_t worker_count =
        std::min({count, available_workers, kMaximumWorkers});

    if (worker_count == 1) {
        for (std::size_t index = 0; index < count; ++index) {
            function(index);
        }
        return;
    }

    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const std::size_t begin = count * worker_id / worker_count;
        const std::size_t end = count * (worker_id + 1) / worker_count;

        workers.emplace_back([begin, end, &function, &failure, &failure_mutex] {
            try {
                for (std::size_t index = begin; index < end; ++index) {
                    function(index);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (failure == nullptr) {
                    failure = std::current_exception();
                }
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}

}  // namespace minifaiss::detail
