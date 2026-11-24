// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/h264_packet.h"
#include "video_core/host1x/nx2_video_decoder.h"

extern "C" {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <libswscale/swscale.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

namespace Tegra::Host1x {

namespace {

std::optional<NvdecCommon::VideoCodec> CodecFromMime(std::string_view mime) {
    if (mime.find("vp9") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP9;
    }
    if (mime.find("vp8") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP8;
    }
    // "video/av01" or "video/av1". FFmpeg decodes AV1 with libdav1d.
    if (mime.find("av01") != std::string_view::npos || mime.find("av1") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::AV1;
    }
    if (mime.find("avc") != std::string_view::npos || mime.find("264") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::H264;
    }
    return std::nullopt;
}

} // namespace

bool IsH264Stream(std::span<const u8> data) {
    if (data.size() >= 4) {
        if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
            const u8 nal_type = data[3] & 0x1F;
            if ((data[3] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 5) {
        if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
        const u32 nalu_len = (static_cast<u32>(data[0]) << 24) |
                             (static_cast<u32>(data[1]) << 16) |
                             (static_cast<u32>(data[2]) << 8) |
                             static_cast<u32>(data[3]);
        if (nalu_len > 0 && nalu_len <= data.size() - 4) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 7) {
        if (data[0] == 1 && (data[4] & 0xFC) == 0xFC && (data[5] & 0xE0) == 0xE0 && (data[5] & 0x1F) > 0) {
            return true;
        }
    }
    return false;
}

struct Nx2VideoDecoder::Impl {
    explicit Impl(CodecBackend backend_) : backend{std::move(backend_)} {}

    const CodecBackend backend;
    std::string name;
    std::optional<FFmpeg::Decoder> decoder;
    std::optional<FFmpeg::HardwareContext> hardware;
    std::unique_ptr<FFmpeg::DecoderContext> context;
    SwsContext* scaler{};
    bool open{};
    bool on_gpu{};
    bool received{};
    H264PacketConverter h264_packets;
    /// FFmpeg normally propagates pts. Some hardware decoders lose theirs, so use the earliest
    /// pending timestamp in that case.
    std::multiset<u64> pending_pts;
    std::vector<std::pair<std::vector<u8>, u64>> gpu_packets;
    bool gpu_packets_lost{};
    std::vector<u8> packet; ///< FFmpeg reads up to AV_INPUT_BUFFER_PADDING_SIZE past the data

    ~Impl() {
        sws_freeContext(scaler);
        context.reset();
        hardware.reset();
    }

    bool OpenGpu() {
        context = std::make_unique<FFmpeg::DecoderContext>(*decoder);
        hardware.emplace();
        on_gpu = backend.initialize_hardware(*hardware, *context, *decoder);
        return backend.open(*context, *decoder, on_gpu);
    }

    std::unique_ptr<FFmpeg::DecoderContext> OpenCpu() const {
        auto cpu = std::make_unique<FFmpeg::DecoderContext>(*decoder);
        if (!backend.open(*cpu, *decoder, false)) {
            return {};
        }
        return cpu;
    }

    void UseCpu(std::unique_ptr<FFmpeg::DecoderContext> cpu) {
        context = std::move(cpu);
        hardware.reset();
        on_gpu = false;
    }

    static bool Rejected(int ret) {
        return ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN);
    }

    int Send(const AVPacket* av_packet) const {
        const int ret = backend.send(*context, av_packet, on_gpu);
        if (Rejected(ret)) {
            LOG_ERROR(HW_GPU, "nx2 video: avcodec_send_packet error {}", ret);
        }
        return ret;
    }

    bool Replay(const AVPacket* unsent_packet, bool drain, std::vector<Nx2Picture>& pictures) {
        for (const auto& [data, pts] : gpu_packets) {
            FFmpeg::Packet replay{
                std::span<const u8>{data.data(), data.size() - AV_INPUT_BUFFER_PADDING_SIZE}};
            replay.GetPacket()->pts = static_cast<s64>(pts);
            pending_pts.insert(pts);
            if (Send(replay.GetPacket()) != 0 || !Receive(pictures)) {
                return false;
            }
        }
        if (unsent_packet) {
            pending_pts.insert(static_cast<u64>(unsent_packet->pts));
            return Send(unsent_packet) == 0 && Receive(pictures);
        }
        return !drain || (!Rejected(Send(nullptr)) && Receive(pictures));
    }

    void FallBackToCpu(const AVPacket* unsent_packet, bool drain, std::vector<Nx2Picture>& pictures) {
        auto cpu = gpu_packets_lost ? nullptr : OpenCpu();
        if (!cpu) {
            return;
        }
        auto gpu = std::exchange(context, std::move(cpu));
        auto gpu_pts = std::exchange(pending_pts, {});
        const size_t count = pictures.size();
        on_gpu = false;
        if (Replay(unsent_packet, drain, pictures)) {
            LOG_WARNING(HW_GPU, "nx2 video: host GPU decoder rejected the stream, decoding on the CPU");
            gpu.reset();
            UseCpu(std::move(context));
            return;
        }
        context = std::move(gpu);
        on_gpu = true;
        pending_pts = std::move(gpu_pts);
        pictures.resize(count);
        received = false;
    }

    bool Receive(std::vector<Nx2Picture>& pictures) {
        while (true) {
            auto [frame, failed] = backend.receive(*context, on_gpu);
            if (!frame) {
                return !failed;
            }
            const AVFrame* f = frame->GetFrame();
            const s64 stamp = f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp
                                                                         : f->pts;
            u64 pts{};
            if (stamp != AV_NOPTS_VALUE) {
                pts = static_cast<u64>(stamp);
                if (const auto it = pending_pts.find(pts); it != pending_pts.end()) {
                    pending_pts.erase(it);
                }
            } else if (!pending_pts.empty()) {
                pts = *pending_pts.begin();
                pending_pts.erase(pending_pts.begin());
            }
            pictures.push_back({static_cast<u32>(f->width), static_cast<u32>(f->height), pts,
                                std::move(frame)});
            received = true;
        }
    }
};

Nx2VideoDecoder::Nx2VideoDecoder(std::string_view mime)
    : Nx2VideoDecoder{mime, {
          .initialize_hardware = [](auto& hardware, auto& context, const auto& decoder) {
              return hardware.InitializeForDecoder(context, decoder);
          },
          .open = [](auto& context, const auto& decoder, bool) { return context.OpenContext(decoder); },
          .send = [](auto& context, const AVPacket* packet, bool) {
              return avcodec_send_packet(context.GetCodecContext(), packet);
          },
          .receive = [](auto& context, bool) {
              auto frame = context.ReceiveFrame();
              return ReceiveResult{std::move(frame), context.ReceiveFailed()};
          },
      }} {}

Nx2VideoDecoder::Nx2VideoDecoder(std::string_view mime, CodecBackend backend)
    : impl{std::make_unique<Impl>(std::move(backend))} {
    impl->name = std::string{mime};
    const auto codec = CodecFromMime(mime);
    if (!codec) {
        LOG_ERROR(HW_GPU, "nx2 video: no decoder for codec \"{}\"", mime);
        return;
    }
    impl->decoder.emplace(*codec);
    if (impl->decoder->GetCodec() == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: FFmpeg has no decoder for \"{}\"", mime);
        return;
    }
    if (Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu) {
        impl->open = impl->OpenGpu();
    }
    if (!impl->open) {
        if (impl->on_gpu) {
            LOG_WARNING(HW_GPU, "nx2 video: host GPU decoder failed to open, decoding on the CPU");
        }
        impl->UseCpu(impl->OpenCpu());
        impl->open = impl->context != nullptr;
    }
    LOG_INFO(HW_GPU, "nx2 video: decoder for \"{}\" {}", mime, impl->open ? "open" : "failed to open");
}

Nx2VideoDecoder::~Nx2VideoDecoder() = default;

bool Nx2VideoDecoder::IsOpen() const {
    return impl->open;
}

const std::string& Nx2VideoDecoder::Name() const {
    return impl->name;
}

void Nx2VideoDecoder::Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
    if (!impl->open || data.empty()) {
        return;
    }
    impl->packet.assign(data.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    std::memcpy(impl->packet.data(), data.data(), data.size());

    // Convert MP4/avcC length-prefixed NALUs to Annex-B start codes for FFmpeg if needed.
    if (impl->name.find("avc") != std::string::npos || impl->name.find("264") != std::string::npos) {
        impl->h264_packets.Convert(std::span<u8>{impl->packet.data(), data.size()});
    }

    FFmpeg::Packet packet{std::span<const u8>{impl->packet.data(), data.size()}};
    packet.GetPacket()->pts = static_cast<s64>(pts);
    const bool waiting = impl->on_gpu && !impl->received;
    const int ret = impl->Send(packet.GetPacket());
    if (ret == 0) {
        // Hidden VP9 frames never come out: keep the set from growing over a long movie.
        if (impl->pending_pts.size() >= 64) {
            impl->pending_pts.erase(impl->pending_pts.begin());
        }
        impl->pending_pts.insert(pts);
        if (waiting && impl->gpu_packets.size() == 64) {
            impl->gpu_packets.clear();
            impl->gpu_packets_lost = true;
        } else if (waiting && !impl->gpu_packets_lost) {
            impl->gpu_packets.emplace_back(impl->packet, pts);
        }
    }
    const bool receive_ok = impl->Receive(pictures);
    if (impl->on_gpu && !impl->received && (Impl::Rejected(ret) || !receive_ok)) {
        impl->FallBackToCpu(ret == 0 ? nullptr : packet.GetPacket(), false, pictures);
    }
    if (impl->received || !impl->on_gpu) {
        impl->gpu_packets.clear();
    }
}

void Nx2VideoDecoder::Drain(std::vector<Nx2Picture>& pictures) {
    if (!impl->open) {
        return;
    }
    const int ret = impl->Send(nullptr);
    const bool receive_ok = impl->Receive(pictures);
    if (impl->on_gpu && !impl->received && (Impl::Rejected(ret) || !receive_ok)) {
        impl->FallBackToCpu(nullptr, true, pictures);
    }
    Reset();
}

void Nx2VideoDecoder::Reset() {
    if (impl->open) {
        avcodec_flush_buffers(impl->context->GetCodecContext());
    }
    impl->pending_pts.clear();
    impl->gpu_packets.clear();
    impl->gpu_packets_lost = false;
    impl->received = false;
}

void Nx2VideoDecoder::Render(const Nx2Picture& picture, Nx2PictureFormat format, u32 width,
                             u32 height, u64 pitch, u64 chroma_offset, std::span<u8> out) {
    const AVFrame* f = picture.frame->GetFrame();
    const u64 needed = format == Nx2PictureFormat::Nv12
                           ? chroma_offset + pitch * ((u64{height} + 1) / 2)
                           : pitch * height;
    if (width == 0 || height == 0 || out.size() < needed) {
        return;
    }
    const AVPixelFormat target = format == Nx2PictureFormat::Nv12    ? AV_PIX_FMT_NV12
                                 : format == Nx2PictureFormat::Rgba8 ? AV_PIX_FMT_RGBA
                                                                     : AV_PIX_FMT_BGRA;
    // Frames copied back from a host GPU decoder are already NV12. Copy them as they are:
    // FFmpeg 6.1's swscale leaves the chroma plane of an NV12 to NV12 copy at zero, which shows
    // the movies green.
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_NV12 &&
        static_cast<u32>(f->width) == width && static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            std::memcpy(out.data() + chroma_offset + y * pitch, f->data[1] + y * f->linesize[1],
                        width);
        }
        return;
    }
    impl->scaler = sws_getCachedContext(impl->scaler, f->width, f->height,
                                        static_cast<AVPixelFormat>(f->format),
                                        static_cast<int>(width), static_cast<int>(height), target,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (impl->scaler == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: cannot convert {} to {}",
                  av_get_pix_fmt_name(static_cast<AVPixelFormat>(f->format)),
                  av_get_pix_fmt_name(target));
        return;
    }
    if (format != Nx2PictureFormat::Nv12) {
        // The stream's matrix and range; untagged HD video is BT.709.
        int space = SWS_CS_ITU601;
        if (f->colorspace == AVCOL_SPC_BT709 ||
            (f->colorspace == AVCOL_SPC_UNSPECIFIED && f->height >= 720)) {
            space = SWS_CS_ITU709;
        } else if (f->colorspace == AVCOL_SPC_BT2020_NCL || f->colorspace == AVCOL_SPC_BT2020_CL) {
            space = SWS_CS_BT2020;
        }
        const int* coefficients = sws_getCoefficients(space);
        sws_setColorspaceDetails(impl->scaler, coefficients, f->color_range == AVCOL_RANGE_JPEG,
                                 coefficients, 1, 0, 1 << 16, 1 << 16);
    }
    std::array<u8*, 4> planes{out.data(), nullptr, nullptr, nullptr};
    std::array<int, 4> strides{static_cast<int>(pitch), 0, 0, 0};
    if (format == Nx2PictureFormat::Nv12) {
        planes[1] = out.data() + chroma_offset;
        strides[1] = static_cast<int>(pitch);
    }
    sws_scale(impl->scaler, f->data, f->linesize, 0, f->height, planes.data(), strides.data());
}

} // namespace Tegra::Host1x
