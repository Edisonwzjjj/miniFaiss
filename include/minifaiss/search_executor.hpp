#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace minifaiss {

class SearchExecutor {
public:
    explicit SearchExecutor(std::size_t worker_count = 0,
                            std::size_t max_pending_tasks = 0);
    ~SearchExecutor();

    SearchExecutor(const SearchExecutor&) = delete;
    SearchExecutor& operator=(const SearchExecutor&) = delete;

    template <typename Function>
    auto submit(Function&& function)
        -> std::future<std::invoke_result_t<std::decay_t<Function>&>> {
        using Result = std::invoke_result_t<std::decay_t<Function>&>;

        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::forward<Function>(function));
        std::future<Result> future = task->get_future();

        enqueue([task] { (*task)(); });
        return future;
    }

    void shutdown();

private:
    void enqueue(std::function<void()> task);
    void worker_loop();

    std::mutex mutex_;
    std::condition_variable task_available_;
    std::condition_variable slot_available_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    std::size_t max_pending_tasks_;
    bool accepting_ = true;
    bool shutdown_requested_ = false;
};

}  // namespace minifaiss
