// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <array>
#include <optional>
#include <span>
#include <utility>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/texture_cache/types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/texture_cache/accelerated_swizzle.h"
#include "video_core/texture_cache/plain_unswizzle.h"

namespace VideoCommon {
struct SwizzleParameters;
}

namespace Vulkan {

using VideoCommon::Accelerated::BlockLinearSwizzle3DParams;

class Device;
class StagingBufferPool;
class Scheduler;
class Image;
struct StagingBufferRef;

class ComputePass {
public:
    explicit ComputePass(const Device& device, Scheduler& scheduler, DescriptorPool& descriptor_pool,
                         vk::Span<VkDescriptorSetLayoutBinding> bindings,
                         vk::Span<VkDescriptorUpdateTemplateEntry> templates,
                         const DescriptorBankInfo& bank_info,
                         vk::Span<VkPushConstantRange> push_constants, std::span<const u32> code,
                         std::optional<u32> optional_subgroup_size = std::nullopt);
    ~ComputePass();

protected:
    /// A compute pipeline over this pass's module. `spec` fixes the module's specialisation
    /// constants; the pipeline the constructor builds passes none.
    vk::Pipeline CreatePipeline(const VkSpecializationInfo* spec = nullptr);

    const Device& device;
    vk::DescriptorUpdateTemplate descriptor_template;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;

private:
    std::optional<u32> required_subgroup_size;
    vk::ShaderModule module;
};

class VertexRepackPass final : public ComputePass {
public:
    VertexRepackPass(const Device& device, Scheduler& scheduler, DescriptorPool& descriptor_pool,
                     StagingBufferPool& staging_pool, ComputePassDescriptorQueue& descriptor_queue);
    /// Returns nothing when the range exceeds the device limits or the source buffer; the binding
    /// is then left as is.
    std::optional<std::pair<VkBuffer, VkDeviceSize>> Assemble(VkBuffer source, u32 offset, u32 size,
                                                              u64 buffer_size);

private:
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;
    ComputePassDescriptorQueue& descriptor_queue;
};

class PlainUnswizzlePass final : public ComputePass {
public:
    PlainUnswizzlePass(const Device& device, Scheduler& scheduler, DescriptorPool& descriptor_pool,
                      ComputePassDescriptorQueue& descriptor_queue);
    void Dispatch(VkBuffer source, VkDeviceSize offset, VkBuffer destination,
                  const VideoCommon::PlainUnswizzleParameters& parameters, u32 guard_bytes = 0);
private:
    Scheduler& scheduler;
    ComputePassDescriptorQueue& descriptor_queue;
};

class Uint8Pass final : public ComputePass {
public:
    explicit Uint8Pass(const Device& device_, Scheduler& scheduler_,
                       DescriptorPool& descriptor_pool_, StagingBufferPool& staging_buffer_pool_,
                       ComputePassDescriptorQueue& compute_pass_descriptor_queue_);
    ~Uint8Pass();

    /// Assemble uint8 indices into an uint16 index buffer
    /// Returns a pair with the staging buffer, and the offset where the assembled data is
    std::pair<VkBuffer, VkDeviceSize> Assemble(u32 num_vertices, VkBuffer src_buffer,
                                               u32 src_offset);

private:
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

class QuadIndexedPass final : public ComputePass {
public:
    explicit QuadIndexedPass(const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             StagingBufferPool& staging_buffer_pool_,
                             ComputePassDescriptorQueue& compute_pass_descriptor_queue_);
    ~QuadIndexedPass();

    std::pair<VkBuffer, VkDeviceSize> Assemble(
        Tegra::Engines::Maxwell3D::Regs::IndexFormat index_format, u32 num_vertices,
        u32 base_vertex, VkBuffer src_buffer, u32 src_offset, bool is_strip);

private:
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

class ConditionalRenderingResolvePass final : public ComputePass {
public:
    explicit ConditionalRenderingResolvePass(
        const Device& device_, Scheduler& scheduler_, DescriptorPool& descriptor_pool_,
        ComputePassDescriptorQueue& compute_pass_descriptor_queue_);

    void Resolve(VkBuffer dst_buffer, VkBuffer src_buffer, u32 src_offset, bool compare_to_zero);

private:
    Scheduler& scheduler;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

class QueriesPrefixScanPass final : public ComputePass {
public:
    explicit QueriesPrefixScanPass(const Device& device_, Scheduler& scheduler_,
                                   DescriptorPool& descriptor_pool_,
                                   ComputePassDescriptorQueue& compute_pass_descriptor_queue_);

    void Run(VkBuffer accumulation_buffer, VkBuffer dst_buffer, VkBuffer src_buffer,
             size_t number_of_sums, size_t min_accumulation_limit, size_t max_accumulation_limit);

private:
    Scheduler& scheduler;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

class ASTCDecoderPass final : public ComputePass {
public:
    explicit ASTCDecoderPass(const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             StagingBufferPool& staging_buffer_pool_,
                             ComputePassDescriptorQueue& compute_pass_descriptor_queue_,
                             MemoryAllocator& memory_allocator_);
    ~ASTCDecoderPass();

    void Assemble(Image& image, const StagingBufferRef& map,
                  std::span<const VideoCommon::SwizzleParameters> swizzles);

    /// Decodes only `layers` of `image` (an initialized array); `map` holds them in order.
    void AssembleLayers(Image& image, const StagingBufferRef& map,
                        std::span<const VideoCommon::SwizzleParameters> swizzles,
                        std::span<const u32> layers);

    /// Where a decode writes: an RGBA8 image with a storage view per (host) level.
    struct Target {
        VkImage image;
        VkImageAspectFlags aspect_mask;
        bool initialized;
        std::function<VkImageView(s32 host_level)> level_view;
        bool wait; ///< Wait for the GPU afterwards (the decoded image itself is sampled).
        /// When set, the input holds only these layers (in order), each decoded into
        /// layer_view(host level, layer).
        std::span<const u32> layers{};
        std::function<VkImageView(s32 host_level, u32 layer)> layer_view{};
    };

    /// Decodes `info`'s ASTC data in `map` into `target`. `skip` leading levels are absent on the
    /// host (TrimmedLevels) and `guest_size_bytes` is the full guest size.
    void AssembleInto(const VideoCommon::ImageInfo& info, u32 skip, size_t guest_size_bytes,
                      const Target& target, const StagingBufferRef& map,
                      std::span<const VideoCommon::SwizzleParameters> swizzles);

private:
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
    MemoryAllocator& memory_allocator;
};

/// Compresses an RGBA8 image level to BC1 or BC3 blocks in a buffer (bc_encoder.comp).
class BcEncoderPass final : public ComputePass {
public:
    explicit BcEncoderPass(const Device& device_, Scheduler& scheduler_,
                           DescriptorPool& descriptor_pool_,
                           ComputePassDescriptorQueue& compute_pass_descriptor_queue_);
    ~BcEncoderPass();

    /// Encodes every layer of one level (`source` is its RGBA8 storage view) into `buffer`,
    /// `out_offset` bytes past `buffer_offset`; blocks are row-major, layer after layer.
    void Encode(VkImageView source, u32 width, u32 height, u32 layers, bool bc3, VkBuffer buffer,
                VkDeviceSize buffer_offset, VkDeviceSize buffer_size, VkDeviceSize out_offset);

private:
    Scheduler& scheduler;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

/// The block-linear geometry and the encoder's mode for one level, as astc_bc.comp declares it.
struct AstcBcPushConstants {
    u32 width;
    u32 height;
    u32 blocks_x;         ///< ASTC blocks in a row of the level
    u32 blocks_y;
    u32 srgb;
    u32 block_w;          ///< the ASTC footprint
    u32 block_h;
    u32 recip_w;          ///< ceil(2^18 / block_w): v / block_w is (v * recip_w) >> 18
    u32 recip_h;
    u32 infill_ds;        ///< the weight grid infill's steps, (1024 + block_w / 2) / (block_w - 1)
    u32 infill_dt;
    u32 layer_stride;     ///< block-linear, from MakeBlockLinearSwizzle2DParams
    u32 block_size;
    u32 x_shift;
    u32 block_height;
    u32 block_height_mask;
    u32 native_tiling;
    u32 out_offset;       ///< the level's first output word, in 8-byte units
};
static_assert(sizeof(AstcBcPushConstants) <= 128);

/// ASTC blocks straight into BC1/BC3 blocks in a buffer, one dispatch, no RGBA8 image
/// (astc_bc.comp): the fused replacement for ASTCDecoderPass + BcEncoderPass on the
/// re-compression path.
class AstcBcPass final : public ComputePass {
public:
    explicit AstcBcPass(const Device& device_, Scheduler& scheduler_,
                        DescriptorPool& descriptor_pool_,
                        ComputePassDescriptorQueue& compute_pass_descriptor_queue_);
    ~AstcBcPass();

    /// Every layer of one level: `layers` layers of the block-linear ASTC data at `input` into
    /// the level's BC blocks at `output`, `params.out_offset` words past its start. `kind` is the
    /// bc_encoder.comp MODE_* the blocks take and `refine` whether the refined fit runs; both are
    /// specialisation constants, so each pair is a pipeline of its own and the kinds a level does
    /// not use are not in its shader.
    void Run(const AstcBcPushConstants& params, u32 kind, bool refine, u32 layers, VkBuffer input,
             VkDeviceSize input_offset, VkDeviceSize input_size, VkBuffer output,
             VkDeviceSize output_offset, VkDeviceSize output_size);

private:
    /// The pipeline for a (kind, refine) pair, compiled on first use.
    VkPipeline PipelineFor(u32 kind, bool refine);

    Scheduler& scheduler;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
    /// MODE_BC1 to MODE_BC1_OPAQUE with refine off, then the same with refine on. vk::Pipeline is
    /// an owning handle: it destroys the pipeline when it dies, so these hold it, not the raw
    /// VkPipeline.
    std::array<std::optional<vk::Pipeline>, 10> pipelines;
};

class BlockLinearUnswizzle3DPass final : public ComputePass {
public:
    explicit BlockLinearUnswizzle3DPass(const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             StagingBufferPool& staging_buffer_pool_,
                             ComputePassDescriptorQueue& compute_pass_descriptor_queue_);
    ~BlockLinearUnswizzle3DPass();

    void Unswizzle(Image& image,
                   const StagingBufferRef& swizzled,
                   std::span<const VideoCommon::SwizzleParameters> swizzles,
                   u32 z_start, u32 z_count);

    void UnswizzleChunk(
        Image& image,
        const StagingBufferRef& swizzled,
        const VideoCommon::SwizzleParameters& sw,
        const BlockLinearSwizzle3DParams& params,
        u32 blocks_x, u32 blocks_y,
        u32 z_start, u32 z_count);

private:
    Scheduler& scheduler;
    StagingBufferPool& staging_buffer_pool;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
};

} // namespace Vulkan
