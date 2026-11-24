// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

struct AVPacket;

namespace FFmpeg {
class Decoder;
class DecoderContext;
class Frame;
class HardwareContext;
}

namespace Tegra::Host1x {

/// A picture decoded by Nx2VideoDecoder, in display order.
struct Nx2Picture {
    u32 width{};
    u32 height{};
    u64 pts{}; ///< As submitted with the compressed frame
    std::shared_ptr<FFmpeg::Frame> frame;
};

enum class Nx2PictureFormat : u32 {
    Nv12,
    Rgba8,
    Bgra8,
};

bool IsH264Stream(std::span<const u8> data);

/// The decoder behind the Switch 2 video decoder service (nn::codec::video): FFmpeg decodes the
/// elementary stream the game demuxed, in software or on the host GPU (NVDEC emulation setting).
/// Used from one thread at a time.
class Nx2VideoDecoder {
public:
    /// mime: the codec name nn::codec sends with Configure ("video/vp9", "video/vp8", ...).
    explicit Nx2VideoDecoder(std::string_view mime);
    ~Nx2VideoDecoder();

    bool IsOpen() const;

    const std::string& Name() const;

    void Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures);

    void Drain(std::vector<Nx2Picture>& pictures);

    void Reset();

    void Render(const Nx2Picture& picture, Nx2PictureFormat format, u32 width, u32 height,
                u64 pitch, u64 chroma_offset, std::span<u8> out);

private:
    friend class Nx2VideoDecoderTestAccess;

    struct ReceiveResult {
        std::shared_ptr<FFmpeg::Frame> frame;
        bool failed{};
    };

    struct CodecBackend {
        std::function<bool(FFmpeg::HardwareContext&, FFmpeg::DecoderContext&,
                           const FFmpeg::Decoder&)> initialize_hardware;
        std::function<bool(FFmpeg::DecoderContext&, const FFmpeg::Decoder&, bool)> open;
        std::function<int(FFmpeg::DecoderContext&, const AVPacket*, bool)> send;
        std::function<ReceiveResult(FFmpeg::DecoderContext&, bool)> receive;
    };

    Nx2VideoDecoder(std::string_view mime, CodecBackend backend);

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Tegra::Host1x
