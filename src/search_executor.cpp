#include "minifaiss/search_executor.hpp"

#include <algorithm>

namespace minifaiss {
namespace {

std::size_t resolve_worker_count(std::size_t worker_count) {
    if (worker_count != 0) {
        return worker_count;
    }

    const unsigned int hardware_count = std::thread::hardware_concurrency();
    return hardware_count == 0 ? 1 : static_cast<std::size_t>(hardware_count);
}

}  // namespace

SearchExecutor::SearchExecutor(std::size_t worker_count,
                               std::size_t max_pending_tasks)
    : max_pending_tasks_(max_pending_tasks) {
    worker_count = resolve_worker_count(worker_count);
    if (max_pending_tasks_ == 0) {
        max_pending_tasks_ = worker_count;
    }

    workers_.reserve(worker_count);
    for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        workers_.emplace_back(&SearchExecutor::worker_loop, this);
    }
}

SearchExecutor::~SearchExecutor() { shutdown(); }

void SearchExecutor::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (shutdown_requested_) {
            return;
        }

        accepting_ = false;
        shutdown_requested_ = true;
    }

    task_available_.notify_all();
    slot_available_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void SearchExecutor::enqueue(std::function<void()> task) {
    std::unique_lock lock(mutex_);
    slot_available_.wait(lock, [this] {
        return !accepting_ || tasks_.size() < max_pending_tasks_;
    });

    if (!accepting_) {
        throw std::runtime_error("cannot submit work after executor shutdown");
    }

    tasks_.push(std::move(task));
    task_available_.notify_one();
}

void SearchExecutor::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock lock(mutex_);
            task_available_.wait(lock, [this] {
                return shutdown_requested_ || !tasks_.empty();
            });

            if (tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            slot_available_.notify_one();
        }

        task();
    }
}

}  // namespace minifaiss
