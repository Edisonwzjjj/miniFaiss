#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "search_result.hpp"

namespace minifaiss {

enum class WalOperationType : std::uint8_t {
    Add,
    Delete,
};

struct WalOperation {
    std::uint64_t sequence_number;
    std::uint64_t operation_id;
    WalOperationType type;
    IndexId id;
    std::vector<float> vector;
};

class WriteAheadLog {
public:
    explicit WriteAheadLog(std::filesystem::path path);

    [[nodiscard]] const std::filesystem::path& path() const noexcept;

    void append(const WalOperation& operation);
    [[nodiscard]] std::vector<WalOperation> replay() const;

private:  
    std::filesystem::path path_;
};

}  // namespace minifaiss
