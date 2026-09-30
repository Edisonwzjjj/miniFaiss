#include "minifaiss/write_ahead_log.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

class TestRunner {
public:
    void check(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ++failures_;
        }
    }

    template <typename Function>
    void check_throws_invalid_argument(Function&& function,
                                       const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::invalid_argument&) {
        }
    }

    template <typename Function>
    void check_throws_runtime_error(Function&& function, const char* message) {
        try {
            function();
            check(false, message);
        } catch (const std::runtime_error&) {
        }
    }

    [[nodiscard]] int exit_code() const { return failures_ == 0 ? 0 : 1; }

private:
    int failures_ = 0;
};

std::filesystem::path wal_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void test_missing_wal_replays_empty(TestRunner& tests) {
    const std::filesystem::path path = wal_path("minifaiss_wal_missing.bin");
    std::filesystem::remove(path);

    const minifaiss::WriteAheadLog wal(path);
    tests.check(wal.path() == path, "WAL stores its path");
    tests.check(wal.replay().empty(), "missing WAL replays no operations");
}

void test_append_and_replay_operations(TestRunner& tests) {
    const std::filesystem::path path = wal_path("minifaiss_wal_replay.bin");
    std::filesystem::remove(path);

    minifaiss::WriteAheadLog wal(path);
    wal.append({1, 101, minifaiss::WalOperationType::Add, 7,
                std::vector<float>{1.0F, 2.0F}});
    wal.append({2, 102, minifaiss::WalOperationType::Delete, 7, {}});
    wal.append({3, 103, minifaiss::WalOperationType::Add, 8,
                std::vector<float>{3.0F}});

    const minifaiss::WriteAheadLog reopened_wal(path);
    const std::vector<minifaiss::WalOperation> operations = reopened_wal.replay();
    tests.check(operations.size() == 3, "WAL replays every complete operation");
    tests.check(operations[0].sequence_number == 1 &&
                    operations[0].operation_id == 101 &&
                    operations[0].type == minifaiss::WalOperationType::Add &&
                    operations[0].id == 7 &&
                    operations[0].vector == std::vector<float>{1.0F, 2.0F},
                "WAL restores add operation fields");
    tests.check(operations[1].sequence_number == 2 &&
                    operations[1].operation_id == 102 &&
                    operations[1].type == minifaiss::WalOperationType::Delete &&
                    operations[1].id == 7 && operations[1].vector.empty(),
                "WAL restores delete operation fields");
    tests.check(operations[2].sequence_number == 3 && operations[2].id == 8,
                "WAL preserves append order");

    std::filesystem::remove(path);
}

void test_append_rejects_invalid_operations(TestRunner& tests) {
    const std::filesystem::path path = wal_path("minifaiss_wal_invalid.bin");
    std::filesystem::remove(path);
    minifaiss::WriteAheadLog wal(path);

    tests.check_throws_invalid_argument(
        [&wal] {
            wal.append({1, 1, minifaiss::WalOperationType::Add, 0, {}});
        },
        "WAL rejects add with empty vector");
    tests.check_throws_invalid_argument(
        [&wal] {
            wal.append({1, 1, minifaiss::WalOperationType::Delete, 0,
                        std::vector<float>{1.0F}});
        },
        "WAL rejects delete with vector payload");
    tests.check_throws_invalid_argument(
        [&wal] {
            wal.append({1, 1, minifaiss::WalOperationType::Add, 0,
                        std::vector<float>{
                            std::numeric_limits<float>::quiet_NaN(),
                        }});
        },
        "WAL rejects non-finite add vector");
    tests.check(wal.replay().empty(),
                "invalid WAL operations do not write partial records");

    std::filesystem::remove(path);
}

void test_replay_ignores_truncated_tail(TestRunner& tests) {
    const std::filesystem::path path = wal_path("minifaiss_wal_truncated.bin");
    std::filesystem::remove(path);
    minifaiss::WriteAheadLog wal(path);
    wal.append({1, 1, minifaiss::WalOperationType::Add, 0,
                std::vector<float>{1.0F, 2.0F}});
    const std::uintmax_t first_record_size = std::filesystem::file_size(path);
    wal.append({2, 2, minifaiss::WalOperationType::Add, 1,
                std::vector<float>{3.0F, 4.0F}});
    std::filesystem::resize_file(path, first_record_size + 30U);

    const std::vector<minifaiss::WalOperation> operations = wal.replay();
    tests.check(operations.size() == 1 && operations[0].sequence_number == 1,
                "WAL ignores a truncated tail record");

    std::filesystem::remove(path);
    wal.append({3, 3, minifaiss::WalOperationType::Delete, 0, {}});
    std::ofstream output(path, std::ios::binary | std::ios::app);
    const std::array<char, 3> prefix_tail = {'x', 'y', 'z'};
    output.write(prefix_tail.data(), static_cast<std::streamsize>(prefix_tail.size()));
    output.close();
    const std::vector<minifaiss::WalOperation> with_prefix_tail = wal.replay();
    tests.check(with_prefix_tail.size() == 1,
                "WAL ignores a truncated tail prefix");

    std::filesystem::remove(path);
}

void write_u64(std::vector<std::uint8_t>& bytes, std::size_t offset,
               std::uint64_t value) {
    for (std::size_t byte_id = 0; byte_id < 8; ++byte_id) {
        bytes[offset + byte_id] = static_cast<std::uint8_t>(value & 0xFFU);
        value >>= 8U;
    }
}

void test_replay_rejects_complete_invalid_record(TestRunner& tests) {
    const std::filesystem::path path = wal_path("minifaiss_wal_corrupt.bin");
    std::filesystem::remove(path);
    std::array<std::uint8_t, 25> invalid_record{};
    invalid_record[16] = 99U;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(invalid_record.data()),
                 static_cast<std::streamsize>(invalid_record.size()));
    output.close();

    const minifaiss::WriteAheadLog wal(path);
    tests.check_throws_runtime_error(
        [&wal] { (void)wal.replay(); },
        "WAL rejects a complete record with an invalid type");

    std::vector<std::uint8_t> empty_add_record(41, 0);
    empty_add_record[16] = static_cast<std::uint8_t>(
        minifaiss::WalOperationType::Add);
    write_u64(empty_add_record, 17, 16);
    write_u64(empty_add_record, 25, 7);
    write_u64(empty_add_record, 33, 0);
    output.open(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(empty_add_record.data()),
                 static_cast<std::streamsize>(empty_add_record.size()));
    output.close();
    tests.check_throws_runtime_error(
        [&wal] { (void)wal.replay(); },
        "WAL rejects a complete add record with an empty vector");

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    TestRunner tests;
    test_missing_wal_replays_empty(tests);
    test_append_and_replay_operations(tests);
    test_append_rejects_invalid_operations(tests);
    test_replay_ignores_truncated_tail(tests);
    test_replay_rejects_complete_invalid_record(tests);

    if (tests.exit_code() == 0) {
        std::cout << "All WriteAheadLog tests passed.\n";
    }
    return tests.exit_code();
}
