#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint64_t u64(const std::byte* bytes) {
    std::uint64_t result = 0;
    for (int i = 0; i < 8; ++i) {
        result |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return result;
}

// Independently inspect the stored directory, then compare complete projected payloads against
// their source file ranges. This catches descriptor-only remapping turning into a transpose,
// a copied alias selecting a wrong parent, or changes to packed codes/scales.
void verify_payloads(const std::filesystem::path& path, const Reader& reader) {
    std::ifstream source(path, std::ios::binary);
    std::array<std::byte, 32> header;
    source.read(reinterpret_cast<char*>(header.data()), header.size());
    require(source.good() && header[7] == std::byte{3}, "test requires an official v3 artifact");
    std::string directory_bytes(u64(header.data() + 8), '\0');
    source.read(directory_bytes.data(), directory_bytes.size());
    require(source.good(), "cannot read v3 directory");
    const Json directory = Json::parse(directory_bytes);
    std::vector<char> buffer(256 * 1024);
    int compared = 0;
    for (const auto& descriptor : reader.objects()) {
        const std::string name(object_name(descriptor));
        if (!name.starts_with("dflash/")) { continue; }
        std::string logical = "dflash2/" + name.substr(7);
        if (logical.ends_with("attention/query_key_value")) {
            logical.replace(logical.size() - 15, 15, "query");
        } else if (logical.ends_with("mlp/gate_up")) {
            logical.replace(logical.size() - 7, 7, "gate");
        }
        const auto& binding = directory.at("bindings").at(logical);
        const std::string id = binding.contains("object")
                                   ? binding.at("object").get<std::string>()
                                   : binding.at("parts")[0].at("object").get<std::string>();
        const auto stored = std::find_if(directory.at("objects").begin(),
                                        directory.at("objects").end(),
            [&](const Json& object) { return object.at("id") == id; });
        require(stored != directory.at("objects").end(), name + ": missing source object");
        const auto payload = reader.payload(descriptor).data;
        require(payload.size() == stored->at("bytes").get<std::uint64_t>(),
                name + ": payload size changed");
        source.seekg(reader.payload_offset() + stored->at("offset").get<std::uint64_t>());
        for (std::size_t offset = 0; offset < payload.size();) {
            const std::size_t bytes = std::min(buffer.size(), payload.size() - offset);
            source.read(buffer.data(), bytes);
            require(source.good() && std::equal(buffer.data(), buffer.data() + bytes,
                    reinterpret_cast<const char*>(payload.data() + offset)),
                    name + ": stored payload was changed");
            offset += bytes;
        }
        ++compared;
    }
    require(compared == 66, "DFlash2 projected inventory is incomplete");
}

void verify_plans(const Reader& reader) {
    for (bool enabled : {false, true}) {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38Nvfp4,
            {.speculative = enabled ? ninfer::SpeculativeBackend::DFlash
                                    : ninfer::SpeculativeBackend::None}, 2);
        require(plan.bindings.dflash.projection_format == NumericFormat::W8G32_F16S,
                "v3 DFlash projection format changed");
        std::array<int, 2> resident{};
        for (const auto& placement : plan.materialization.device_objects) {
            const auto name = object_name(binder.descriptor(placement.object));
            if (name.starts_with("dflash/")) { ++resident.at(placement.device); }
            require(!name.starts_with("mtp/"), "DFlash/ordinary startup made MTP resident");
        }
        require(resident == (enabled ? std::array{66, 66} : std::array{0, 0}),
                "DFlash2 startup materialization did not follow selected capability");
        for (const auto& layer : plan.bindings.dflash.layers) {
            const auto& qkv = std::get<TensorDescriptor>(binder.descriptor(layer.query_key_value));
            require(qkv.format == NumericFormat::W8G32_F16S &&
                    qkv.shape == std::vector<std::uint64_t>{6144, 5120},
                    "DFlash2 QKV binding is not the full packed matrix");
            const auto& base = std::get<TensorDescriptor>(
                binder.descriptor(layer.attention_conv_base_kernel));
            require(base.format == NumericFormat::BF16 &&
                    base.shape == std::vector<std::uint64_t>{5120, 2, 2},
                    "DFlash2 convolution is not the ne0-fast channel/tap/side view");
        }
    }
}

} // namespace

int main() {
    const char* configured = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
    const std::filesystem::path path = configured && *configured
        ? std::filesystem::path(configured)
        : std::filesystem::path(NINFER_SOURCE_DIR) / "out/qwen3_8_27b_nvfp4.ninfer";
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: official Qwen3.8 NVFP4 v3 artifact unavailable at " << path << '\n';
        return 77;
    }
    try {
        Reader reader(path);
        verify_payloads(path, reader);
        verify_plans(reader);
        std::cout << "OK DFlash2 v3 exact payloads and startup capabilities\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
