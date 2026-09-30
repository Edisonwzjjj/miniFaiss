#include "minifaiss/write_ahead_log.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "binary_io.hpp"

namespace minifaiss {
namespace {

constexpr std::size_t kRecordPrefixSize = 25;
constexpr std::size_t kAddPayloadPrefixSize = 16;
constexpr std::size_t kDeletePayloadSize = 8;

std::size_t as_index_id(std::uint64_t value) {
    if (value > std::numeric_limits<IndexId>::max()) {
        throw std::runtime_error("WAL ID is too large");
    }
    return value;
}

void write_operation_payload(const WalOperation& operation,
                             detail::BinaryWriter& payload) {
    switch (operation.type) {
        case WalOperationType::Add:
            if (operation.vector.empty()) {
                throw std::invalid_argument("WAL add vector must not be empty");
            }
            payload.write_u64(operation.id);
            payload.write_u64(operation.vector.size());
            for (const float value : operation.vector) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "WAL add vector must contain only finite values");
                }
                payload.write_float(value);
            }
            return;
        case WalOperationType::Delete:
            if (!operation.vector.empty()) {
                throw std::invalid_argument("WAL delete must not contain a vector");
            }
            payload.write_u64(operation.id);
            return;
    }
    throw std::invalid_argument("WAL operation type is invalid");
}

WalOperation read_add_operation(std::uint64_t sequence_number,
                                std::uint64_t operation_id,
                                const std::vector<std::uint8_t>& payload_bytes) {
    if (payload_bytes.size() < kAddPayloadPrefixSize) {
        throw std::runtime_error("WAL add payload is invalid");
    }
    detail::BinaryReader payload(payload_bytes);
    const IndexId id = as_index_id(payload.read_u64());
    const std::size_t vector_count = detail::as_size(payload.read_u64());
    if (vector_count == 0) {
        throw std::runtime_error("WAL add vector must not be empty");
    }
    if (vector_count >
        (payload_bytes.size() - kAddPayloadPrefixSize) / sizeof(float)) {
        throw std::runtime_error("WAL add vector length is invalid");
    }
    const std::size_t vector_bytes = detail::checked_product(
        vector_count, sizeof(float));
    if (payload_bytes.size() != kAddPayloadPrefixSize + vector_bytes) {
        throw std::runtime_error("WAL add payload length is invalid");
    }

    WalOperation operation{sequence_number, operation_id, WalOperationType::Add,
                           id, std::vector<float>(vector_count)};
    for (float& value : operation.vector) {
        value = payload.read_float();
        if (!std::isfinite(value)) {
            throw std::runtime_error("WAL add vector contains non-finite value");
        }
    }
    if (!payload.empty()) {
        throw std::runtime_error("WAL add payload has trailing bytes");
    }
    return operation;
}

WalOperation read_delete_operation(
    std::uint64_t sequence_number, std::uint64_t operation_id,
    const std::vector<std::uint8_t>& payload_bytes) {
    if (payload_bytes.size() != kDeletePayloadSize) {
        throw std::runtime_error("WAL delete payload length is invalid");
    }
    detail::BinaryReader payload(payload_bytes);
    const IndexId id = as_index_id(payload.read_u64());
    return {sequence_number, operation_id, WalOperationType::Delete, id, {}};
}

}  // namespace

WriteAheadLog::WriteAheadLog(std::filesystem::path path) : path_(std::move(path)) {}

const std::filesystem::path& WriteAheadLog::path() const noexcept {
    return path_;
}

void WriteAheadLog::append(const WalOperation& operation) {
    detail::BinaryWriter payload;
    write_operation_payload(operation, payload);

    detail::BinaryWriter record;
    record.write_u64(operation.sequence_number);
    record.write_u64(operation.operation_id);
    record.write_u8(static_cast<std::uint8_t>(operation.type));
    record.write_u64(payload.bytes().size());
    record.write_bytes(payload.bytes().data(), payload.bytes().size());

    std::ofstream output(path_, std::ios::binary | std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot open WAL for writing");
    }
    output.write(reinterpret_cast<const char*>(record.bytes().data()),
                 static_cast<std::streamsize>(record.bytes().size()));
    output.flush();
    if (!output) {
        throw std::runtime_error("cannot write WAL");
    }
}

std::vector<WalOperation> WriteAheadLog::replay() const {
    if (!std::filesystem::exists(path_)) {
        return {};
    }

    std::ifstream input(path_, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open WAL for reading");
    }
    const std::streamoff end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine WAL size");
    }
    input.seekg(0);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.read(reinterpret_cast<char*>(bytes.data()), end);
    if (!input && !bytes.empty()) {
        throw std::runtime_error("cannot read WAL");
    }

    std::vector<WalOperation> operations;
    std::size_t offset = 0;
    while (bytes.size() - offset >= kRecordPrefixSize) {
        std::vector<std::uint8_t> prefix_bytes;
        prefix_bytes.reserve(kRecordPrefixSize);
        for (std::size_t byte_id = 0; byte_id < kRecordPrefixSize; ++byte_id) {
            prefix_bytes.push_back(bytes[offset + byte_id]);
        }
        detail::BinaryReader prefix(prefix_bytes);
        const std::uint64_t sequence_number = prefix.read_u64();
        const std::uint64_t operation_id = prefix.read_u64();
        const std::uint8_t type_value = prefix.read_u8();
        const std::size_t payload_size = detail::as_size(prefix.read_u64());
        offset += kRecordPrefixSize;

        if (payload_size > bytes.size() - offset) {
            break;
        }
        std::vector<std::uint8_t> payload;
        payload.reserve(payload_size);
        for (std::size_t byte_id = 0; byte_id < payload_size; ++byte_id) {
            payload.push_back(bytes[offset + byte_id]);
        }
        offset += payload_size;

        switch (type_value) {
            case static_cast<std::uint8_t>(WalOperationType::Add):
                operations.push_back(read_add_operation(sequence_number,
                                                        operation_id, payload));
                break;
            case static_cast<std::uint8_t>(WalOperationType::Delete):
                operations.push_back(read_delete_operation(sequence_number,
                                                           operation_id, payload));
                break;
            default:
                throw std::runtime_error("WAL operation type is invalid");
        }
    }
    return operations;
}

}  // namespace minifaiss
