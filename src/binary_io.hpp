#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace minifaiss::detail {

enum class IndexFileType : std::uint32_t {
    kFlatIP = 1,
    kIVFFlat = 2,
    kIVFPQ = 3,
    kHNSWFlat = 4,
};

class BinaryWriter {
public:
    void write_u8(std::uint8_t value) { bytes_.push_back(value); }

    void write_u32(std::uint32_t value) {
        for (std::size_t byte_id = 0; byte_id < 4; ++byte_id) {
            bytes_.push_back(static_cast<std::uint8_t>(value & 0xFFU));
            value >>= 8U;
        }
    }

    void write_u64(std::uint64_t value) {
        for (std::size_t byte_id = 0; byte_id < 8; ++byte_id) {
            bytes_.push_back(static_cast<std::uint8_t>(value & 0xFFU));
            value >>= 8U;
        }
    }

    void write_float(float value) { write_u32(std::bit_cast<std::uint32_t>(value)); }

    void write_bytes(const std::uint8_t* values, std::size_t count) {
        bytes_.insert(bytes_.end(), values, values + count);
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept {
        return bytes_;
    }

private:
    std::vector<std::uint8_t> bytes_;
};

class BinaryReader {
public:
    explicit BinaryReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t read_u8() {
        require(1);
        return bytes_[offset_++];
    }

    [[nodiscard]] std::uint32_t read_u32() {
        std::uint32_t value = 0;
        for (std::size_t byte_id = 0; byte_id < 4; ++byte_id) {
            value |= static_cast<std::uint32_t>(read_u8()) << (byte_id * 8U);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t read_u64() {
        std::uint64_t value = 0;
        for (std::size_t byte_id = 0; byte_id < 8; ++byte_id) {
            value |= static_cast<std::uint64_t>(read_u8()) << (byte_id * 8U);
        }
        return value;
    }

    [[nodiscard]] float read_float() {
        return std::bit_cast<float>(read_u32());
    }

    void read_bytes(std::uint8_t* destination, std::size_t count) {
        require(count);
        for (std::size_t byte_id = 0; byte_id < count; ++byte_id) {
            destination[byte_id] = bytes_[offset_ + byte_id];
        }
        offset_ += count;
    }

    [[nodiscard]] bool empty() const noexcept { return offset_ == bytes_.size(); }

private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error("index file is truncated");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

inline std::size_t as_size(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("index file count is too large");
    }
    return static_cast<std::size_t>(value);
}

inline std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::runtime_error("index file count overflows");
    }
    return lhs * rhs;
}

inline void write_index_file(const std::filesystem::path& path,
                             IndexFileType type,
                             const BinaryWriter& payload) {
    constexpr std::uint8_t magic[] = {'M', 'F', 'I', 'D', 'X', 0, 0, 0};
    constexpr std::uint32_t version = 1;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open index file for writing");
    }

    BinaryWriter header;
    header.write_bytes(magic, sizeof(magic));
    header.write_u32(version);
    header.write_u32(static_cast<std::uint32_t>(type));
    header.write_u64(payload.bytes().size());
    output.write(reinterpret_cast<const char*>(header.bytes().data()),
                 static_cast<std::streamsize>(header.bytes().size()));
    output.write(reinterpret_cast<const char*>(payload.bytes().data()),
                 static_cast<std::streamsize>(payload.bytes().size()));
    if (!output) {
        throw std::runtime_error("cannot write index file");
    }
}

inline std::vector<std::uint8_t> read_index_file(
    const std::filesystem::path& path, IndexFileType expected_type) {
    constexpr std::uint8_t magic[] = {'M', 'F', 'I', 'D', 'X', 0, 0, 0};

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open index file for reading");
    }
    const std::streamoff end = input.tellg();
    if (end < 24) {
        throw std::runtime_error("index file header is truncated");
    }
    input.seekg(0);

    std::vector<std::uint8_t> file_bytes(static_cast<std::size_t>(end));
    input.read(reinterpret_cast<char*>(file_bytes.data()), end);
    if (!input) {
        throw std::runtime_error("cannot read index file");
    }

    BinaryReader file_reader(file_bytes);
    for (const std::uint8_t expected : magic) {
        if (file_reader.read_u8() != expected) {
            throw std::runtime_error("index file magic is invalid");
        }
    }
    if (file_reader.read_u32() != 1) {
        throw std::runtime_error("index file version is unsupported");
    }
    if (file_reader.read_u32() != static_cast<std::uint32_t>(expected_type)) {
        throw std::runtime_error("index file type is invalid");
    }
    const std::size_t payload_size = as_size(file_reader.read_u64());
    if (payload_size != file_bytes.size() - 24) {
        throw std::runtime_error("index file payload size is invalid");
    }

    std::vector<std::uint8_t> payload(payload_size);
    file_reader.read_bytes(payload.data(), payload.size());
    if (!file_reader.empty()) {
        throw std::runtime_error("index file has trailing bytes");
    }
    return payload;
}

}  // namespace minifaiss::detail
