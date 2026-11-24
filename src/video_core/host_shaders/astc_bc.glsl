// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* The fused pass's body, included by astc_bc.comp: ASTC blocks straight into BC1, BC3, BC4 or BC5
   blocks (a specialisation constant, ASTC_BC_KIND), in one dispatch, with no RGBA8 image in
   between.

   The two-pass path (TranscodeAstcToBcn with astc_decoder.comp and bc_encoder.comp) decodes a
   level into a temporary RGBA8 image and reads it back to compress it: every texel leaves the
   shader and comes back, four bytes each way, and the image holds four times the bytes of the
   BC3 blocks it exists to make (measured on MKW: 236-851 MiB resident, 74-657 MiB/s through it).

   A block's sixteen texels lie in one ASTC block where the footprint is 4, 8 or 12 a side, else
   in up to two each way. A block decodes once for each 4x4 block over it, at worst nine times for
   12x12, where a decoder shared by the workgroup would decode it once: ScarletNX measured the
   sharing, and its shared memory kept workgroups out of the encoder, which is nearly all of this
   shader's time (7 to 26% slower than the two dispatches on a GT 1030). For a footprint of whole
   4x4 blocks the decode is not repeated at all: PerAstcBlock makes an invocation an ASTC block and
   the blocks inside it one after another, a 1024x1024 level of 8x8 in 2.1 ms against 5.2.

   A block past the level's edge repeats its last texels, as the two-pass encoder does: padding
   adds no colour to the line. */

/* the invocation's texels, packed RGBA8 so they cost 16 registers and not 64, and whether any of
   them has alpha below a half (for punch-through) */
uint px[16];
bool any_transparent;

/* The byte offset of block (ax, ay) of layer z in the upload's block-linear staging buffer: the
   same arithmetic astc_decoder.comp's main() does for the block it was dispatched for. */
uvec4 AstcBlock(uint ax, uint ay, uint z) {
    const uvec2 pos = uvec2(ax << 4u, ay);
    uint swizzle = ((pos.x & 32u) << 3u) | ((pos.y & 6u) << 5u) | ((pos.x & 16u) << 1u) |
                   ((pos.y & 1u) << 4u) | (pos.x & 15u);
    if (native_tiling != 0u) {
        // Switch 2 GOBs swap offset bits 5 and 6 (see block_linear_unswizzle_2d.comp).
        const uint difference = ((swizzle >> 5u) ^ (swizzle >> 6u)) & 1u;
        swizzle ^= (difference << 5u) | (difference << 6u);
    }
    const uint block_y = pos.y >> 3u;
    uint offset = 0u;
    offset += z * layer_stride;
    offset += (block_y >> block_height) * block_size;
    offset += (block_y & block_height_mask) << 9u;
    offset += (pos.x >> 6u) << x_shift;
    offset += swizzle;
    return astc_data[offset / 16u];
}

vec4 UnpackTexel(uint word) {
    return vec4(uvec4(word & 0xFFu, (word >> 8u) & 0xFFu, (word >> 16u) & 0xFFu, word >> 24u)) /
           255.0;
}

/* the i-th texel of the block, from the packed words: what bcn_encode.glsl's encoders read */
vec4 Texel(int i) {
    return UnpackTexel(px[i]);
}

/* the texel coordinates of block (bx, by) inside the level, its last row and column repeated */
void Bounds(uint bx, uint by, out uint x0, out uint y0, out uint x1, out uint y1) {
    x0 = bx * 4u;
    y0 = by * 4u;
    x1 = min(x0 + 3u, width - 1u);
    y1 = min(y0 + 3u, height - 1u);
}

/* the texels inside the level, a bit a texel, which the refined fit measures its error on */
uint InsideMask(uint bx, uint by) {
    const uvec2 inside =
        uvec2(clamp(ivec2(width, height) - ivec2(bx * 4u, by * 4u), ivec2(0), ivec2(4)));
    const uint row = (1u << inside.x) - 1u;
    return row * (((1u << (4u * inside.y)) - 1u) / 15u); // the row in each nibble
}

/* The output kind and whether the refined fit runs, fixed when the pipeline is made (AstcBcPass
   gives the driver both as specialisation constants): a kind a level does not use and the refined
   path leave its shader entirely rather than sitting behind a branch, which is most of its size
   and a good part of what keeps it out of the SIMD's register budget. bc_encoder.comp's MODE_*. */
layout(constant_id = 0) const uint ASTC_BC_KIND = MODE_BC1;
/* REFINE, as bc_encoder.comp: the encoder searches for the fit instead of taking the box fit */
layout(constant_id = 1) const uint ASTC_BC_REFINE = 0u;

/* The block at (bx, by) of slice z, its words written where the level's blocks go, cx to a row. */
void Store(uint cx, uint cy, uint bx, uint by, uint z) {
    const uint at = (z * cy + by) * cx + bx;
    const uint kind = ASTC_BC_KIND;
    const bool refine = ASTC_BC_REFINE != 0u;
    const uint real = InsideMask(bx, by);
    if (kind == MODE_BC3) {
        out_blocks[out_offset + at * 2u] = EncodeAlpha();
        out_blocks[out_offset + at * 2u + 1u] =
            refine ? EncodeColorRefined(real) : EncodeColor(false);
    } else if (kind == MODE_BC4) {
        out_blocks[out_offset + at] = EncodeRed();
    } else if (kind == MODE_BC5) {
        out_blocks[out_offset + at * 2u] = EncodeRed();
        out_blocks[out_offset + at * 2u + 1u] = EncodeAlpha();
    } else {
        const bool punch_through = kind != MODE_BC1_OPAQUE && any_transparent;
        out_blocks[out_offset + at] = refine && !punch_through
                                          ? EncodeColorRefined(real)
                                          : EncodeColor(punch_through);
    }
}

/* Every texel of block (bx, by) from the ASTC blocks they lie in, and whether any is transparent:
   one ASTC block covers it where the footprint is a multiple of four, else up to two each way,
   each decoded once and giving the texels that lie inside it. */
void LoadBlock(uint bx, uint by, uint z) {
    uint x0, y0, x1, y1;
    Bounds(bx, by, x0, y0, x1, y1);
    any_transparent = false;
    if (((block_w | block_h) & 3u) == 0u) {
        const uint ax = (x0 * recip_w) >> 18u;
        const uint ay = (y0 * recip_h) >> 18u;
        const uint left = ax * block_w;
        const uint top = ay * block_h;
        blk = AstcBlock(ax, ay, z);
        DecodeBlock();
        for (uint k = 0u; k < 16u; k++) {
            const uint i = min(x0 + (k & 3u), x1) - left;
            const uint j = min(y0 + (k >> 2u), y1) - top;
            px[k] = SlotTexel(0u, i, j, infill_ds, infill_dt);
            any_transparent = any_transparent || (px[k] >> 24u) < 128u;
        }
        return;
    }
    const uint ax0 = (x0 * recip_w) >> 18u;
    const uint ax1 = (x1 * recip_w) >> 18u;
    const uint ay0 = (y0 * recip_h) >> 18u;
    const uint ay1 = (y1 * recip_h) >> 18u;
    for (uint ay = ay0; ay <= ay1; ay++) {
        for (uint ax = ax0; ax <= ax1; ax++) {
            blk = AstcBlock(ax, ay, z);
            DecodeBlock();
            const uint left = ax * block_w;
            const uint top = ay * block_h;
            for (uint k = 0u; k < 16u; k++) {
                const uint i = min(x0 + (k & 3u), x1) - left;
                const uint j = min(y0 + (k >> 2u), y1) - top;
                if (i < block_w && j < block_h) {
                    px[k] = SlotTexel(0u, i, j, infill_ds, infill_dt);
                    any_transparent = any_transparent || (px[k] >> 24u) < 128u;
                }
            }
        }
    }
}

/* A footprint of whole 4x4 blocks each way (4, 8 or 12 a side): an invocation an ASTC block,
   decoded once, and the blocks inside it made one after another from it, where an invocation a
   block would decode it once for each of them (four times for 8x8, nine for 12x12). The host
   dispatches an invocation an ASTC block for these, which is the common case: the titles here
   ship 8x8. */
void PerAstcBlock(uint cx, uint cy) {
    const uint ax = gl_GlobalInvocationID.x;
    const uint ay = gl_GlobalInvocationID.y;
    const uint z = gl_WorkGroupID.z;
    if (ax >= blocks_x || ay >= blocks_y) {
        return;
    }
    blk = AstcBlock(ax, ay, z);
    DecodeBlock();
    const uint sw = block_w >> 2u;
    const uint sh = block_h >> 2u;
    const uint left = ax * block_w;
    const uint top = ay * block_h;
    for (uint sy = 0u; sy < sh; sy++) {
        for (uint sx = 0u; sx < sw; sx++) {
            const uint bx = ax * sw + sx;
            const uint by = ay * sh + sy;
            if (bx >= cx || by >= cy) {
                continue;
            }
            uint x0, y0, x1, y1;
            Bounds(bx, by, x0, y0, x1, y1);
            any_transparent = false;
            for (uint k = 0u; k < 16u; k++) {
                const uint i = min(x0 + (k & 3u), x1) - left;
                const uint j = min(y0 + (k >> 2u), y1) - top;
                px[k] = SlotTexel(0u, i, j, infill_ds, infill_dt);
                any_transparent = any_transparent || (px[k] >> 24u) < 128u;
            }
            Store(cx, cy, bx, by, z);
        }
    }
}

void main() {
    const uint cx = (width + 3u) >> 2u;
    const uint cy = (height + 3u) >> 2u;
    if (((block_w | block_h) & 3u) == 0u) {
        PerAstcBlock(cx, cy);
        return;
    }
    const uint bx = gl_GlobalInvocationID.x;
    const uint by = gl_GlobalInvocationID.y;
    const uint z = gl_WorkGroupID.z;
    if (bx >= cx || by >= cy) {
        return;
    }
    LoadBlock(bx, by, z);
    Store(cx, cy, bx, by, z);
}
