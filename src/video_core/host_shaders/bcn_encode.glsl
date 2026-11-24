// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This tree's BC1/BC3/BC4/BC5 block encoders, the body bc_encoder.comp and astc_bc.comp share so
   the fused and two-pass paths write the same bytes. The caller declares blocks_x, blocks_y,
   width, height, the output as `uvec2 blocks[]`, and Texel(i) - the i-th of the block's sixteen
   texels. MODE_* and REFINE select the format and the refined endpoints.

   The texels come in through Texel() rather than an array of vec4 because a fused pass already
   holds a decoded ASTC block's state: sixteen vec4 on top of it is 64 registers, where sixteen
   packed RGBA8 words are 16, and registers are what decides how many workgroups the GPU keeps in
   flight. The two-pass encoder defines Texel() over its vec4 array and pays what it always paid. */

/* the i-th texel of the block, RGBA in 0..1: the including shader's */
vec4 Texel(int i);

const uint MODE_BC1 = 0u;        // punch-through where a texel's alpha is below 0.5
const uint MODE_BC3 = 1u;
const uint MODE_BC4 = 2u;        // red
const uint MODE_BC5 = 3u;        // red, and alpha as green
const uint MODE_BC1_OPAQUE = 4u; // four-colour mode for every block
const uint MODE_MASK = 15u;
// Colour endpoints along the principal axis, refit by least squares (EncodeColorRefined).
const uint REFINE = 16u;

uint Pack565(vec3 color) {
    const uvec3 q = uvec3(round(clamp(color, 0.0, 1.0) * vec3(31.0, 63.0, 31.0)));
    return (q.r << 11) | (q.g << 5) | q.b;
}

vec3 Unpack565(uint value) {
    const uvec3 q = uvec3((value >> 11) & 31u, (value >> 5) & 63u, value & 31u);
    const uvec3 e = uvec3((q.r << 3) | (q.r >> 2), (q.g << 2) | (q.g >> 4), (q.b << 3) | (q.b >> 2));
    return vec3(e) / 255.0;
}

float Distance2(vec3 a, vec3 b) {
    const vec3 d = a - b;
    return dot(d, d);
}

// BC1 colour block. With punch_through (BC1 only), texels with alpha < 0.5 use the transparent
// index of the three-colour mode; otherwise the four-colour mode is used (BC3 always decodes so).
uvec2 EncodeColor(bool punch_through) {
    vec3 lo = vec3(1.0);
    vec3 hi = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        if (punch_through && Texel(i).a < 0.5) {
            continue;
        }
        lo = min(lo, Texel(i).rgb);
        hi = max(hi, Texel(i).rgb);
    }
    if (any(greaterThan(lo, hi))) {
        // Every texel transparent: colour0 == colour1 selects three-colour mode, index 3.
        return uvec2(0u, 0xFFFFFFFFu);
    }
    const vec3 center = (lo + hi) * 0.5;
    const vec3 extent = hi - lo;
    const int major = extent.g >= extent.r && extent.g >= extent.b ? 1 : (extent.r >= extent.b ? 0 : 2);
    vec3 covariance = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        if (punch_through && Texel(i).a < 0.5) {
            continue;
        }
        const vec3 d = Texel(i).rgb - center;
        covariance += d * d[major];
    }
    vec3 e0 = hi;
    vec3 e1 = lo;
    for (int c = 0; c < 3; ++c) {
        if (covariance[c] < 0.0) {
            e0[c] = lo[c];
            e1[c] = hi[c];
        }
    }
    const vec3 inset = (e0 - e1) / 16.0;
    e0 -= inset;
    e1 += inset;

    uint q0 = Pack565(e0);
    uint q1 = Pack565(e1);
    // Four-colour mode needs colour0 > colour1, three-colour mode colour0 <= colour1.
    if (punch_through ? q0 > q1 : q0 < q1) {
        const uint t = q0;
        q0 = q1;
        q1 = t;
    }
    if (!punch_through && q0 == q1) {
        return uvec2(q0 | (q1 << 16), 0u);
    }
    vec3 palette[4];
    palette[0] = Unpack565(q0);
    palette[1] = Unpack565(q1);
    if (punch_through) {
        palette[2] = (palette[0] + palette[1]) * 0.5;
        palette[3] = vec3(0.0);
    } else {
        palette[2] = (2.0 * palette[0] + palette[1]) / 3.0;
        palette[3] = (palette[0] + 2.0 * palette[1]) / 3.0;
    }
    const int entries = punch_through ? 3 : 4;
    uint indices = 0u;
    for (int i = 0; i < 16; ++i) {
        uint best = 0u;
        if (punch_through && Texel(i).a < 0.5) {
            best = 3u;
        } else {
            float best_distance = Distance2(Texel(i).rgb, palette[0]);
            for (int p = 1; p < entries; ++p) {
                const float distance = Distance2(Texel(i).rgb, palette[p]);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = uint(p);
                }
            }
        }
        indices |= best << (2 * i);
    }
    return uvec2(q0 | (q1 << 16), indices);
}

// A four-colour block of endpoints `q0` and `q1` (565, in either order): the texels' nearest
// palette entries in `block` (all index 0 when the endpoints are equal), the squared error of the
// texels in `real` (bit i: texel i is inside the level, not an edge block's repeat) returned.
float FitFourColor(uint real, uint q0, uint q1, out uvec2 block) {
    if (q0 < q1) {
        const uint t = q0;
        q0 = q1;
        q1 = t;
    }
    const vec3 c0 = Unpack565(q0);
    const vec3 c1 = Unpack565(q1);
    const vec3 c2 = (2.0 * c0 + c1) / 3.0;
    const vec3 c3 = (c0 + 2.0 * c1) / 3.0;
    float error = 0.0;
    uint indices = 0u;
    for (int i = 0; i < 16; ++i) {
        const vec3 t = Texel(i).rgb;
        float best_distance = Distance2(t, c0);
        uint best = 0u;
        if (q0 != q1) {
            const float d1 = Distance2(t, c1);
            const float d2 = Distance2(t, c2);
            const float d3 = Distance2(t, c3);
            if (d1 < best_distance) {
                best_distance = d1;
                best = 1u;
            }
            if (d2 < best_distance) {
                best_distance = d2;
                best = 2u;
            }
            if (d3 < best_distance) {
                best_distance = d3;
                best = 3u;
            }
        }
        error += ((real >> i) & 1u) != 0u ? best_distance : 0.0;
        indices |= best << (2 * i);
    }
    block = uvec2(q0 | (q1 << 16), indices);
    return error;
}

// Least-squares endpoints for the indices of `block` (index 0: colour0, 1: colour1, 2 and 3 at a
// third and two thirds) over the texels in `real`, as 565 codes; false when the indices don't
// determine them.
bool RefitFourColor(uint real, uvec2 block, out uint q0, out uint q1) {
    vec3 sums = vec3(0.0); // aa, ab, bb
    vec3 ax = vec3(0.0);
    vec3 bx = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        if (((real >> i) & 1u) == 0u) {
            continue;
        }
        const uint index = (block.y >> (2 * i)) & 3u;
        const float b = index == 0u ? 0.0 : (index == 1u ? 1.0 : (index == 2u ? 1.0 / 3.0 : 2.0 / 3.0));
        const float a = 1.0 - b;
        sums += vec3(a * a, a * b, b * b);
        ax += a * Texel(i).rgb;
        bx += b * Texel(i).rgb;
    }
    const float det = sums.x * sums.z - sums.y * sums.y;
    if (abs(det) < 1e-4) {
        return false;
    }
    q0 = Pack565((sums.z * ax - sums.y * bx) / det);
    q1 = Pack565((sums.x * bx - sums.y * ax) / det);
    return true;
}

// Four-colour block with better endpoints than EncodeColor (the endpoint fitting of
// bc7_encoder.comp): the best of EncodeColor's box corners and the extremes of the texels along
// their principal axis (inset by 1/16), two least-squares refits of the best one's indices, then
// single 565 steps of each endpoint channel while they lower the error. Every index is chosen
// from the whole four-entry palette (what the BC7 encoder's neighbour check approximates), and
// the error of the texels inside the level (`real`) decides, so no block ends worse than
// EncodeColor's.
uvec2 EncodeColorRefined(uint real) {
    uvec2 best_block = EncodeColor(false);
    float best =
        FitFourColor(real, best_block.x & 0xFFFFu, best_block.x >> 16, best_block);
    if (best == 0.0) {
        return best_block;
    }
    const float count = float(bitCount(real));
    vec3 mean = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        mean += ((real >> i) & 1u) != 0u ? Texel(i).rgb : vec3(0.0);
    }
    mean /= count;
    vec3 cov0 = vec3(0.0); // rows of the covariance matrix
    vec3 cov1 = vec3(0.0);
    vec3 cov2 = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        const vec3 d = ((real >> i) & 1u) != 0u ? Texel(i).rgb - mean : vec3(0.0);
        cov0 += d * d.r;
        cov1 += d * d.g;
        cov2 += d * d.b;
    }
    // Power iteration from the row of the widest channel.
    vec3 axis = cov0.r >= cov1.g && cov0.r >= cov2.b ? cov0 : (cov1.g >= cov2.b ? cov1 : cov2);
    for (int k = 0; k < 4; ++k) {
        const float length2 = dot(axis, axis);
        if (length2 < 1e-12) {
            break;
        }
        axis = vec3(dot(cov0, axis), dot(cov1, axis), dot(cov2, axis)) * inversesqrt(length2);
    }
    if (dot(axis, axis) >= 1e-12) {
        axis = normalize(axis);
        float t_lo = 1e9;
        float t_hi = -1e9;
        for (int i = 0; i < 16; ++i) {
            if (((real >> i) & 1u) != 0u) {
                const float t = dot(Texel(i).rgb - mean, axis);
                t_lo = min(t_lo, t);
                t_hi = max(t_hi, t);
            }
        }
        const float inset = (t_hi - t_lo) / 16.0;
        uvec2 block;
        const float error = FitFourColor(real, Pack565(mean + axis * (t_hi - inset)),
                                         Pack565(mean + axis * (t_lo + inset)), block);
        if (error < best) {
            best = error;
            best_block = block;
        }
    }
    for (int k = 0; k < 2; ++k) {
        uint q0;
        uint q1;
        if (!RefitFourColor(real, best_block, q0, q1)) {
            break;
        }
        uvec2 block;
        const float error = FitFourColor(real, q0, q1, block);
        if (error >= best) {
            break;
        }
        best = error;
        best_block = block;
    }
    // The refit rounds each channel on its own: try the neighbouring codes.
    for (int pass = 0; pass < 2; ++pass) {
        bool improved = false;
        for (uint c = 0u; c < 6u; ++c) {
            const uint shift = c % 3u == 0u ? 11u : (c % 3u == 1u ? 5u : 0u);
            const uint top = c % 3u == 1u ? 63u : 31u;
            for (int step = -1; step <= 1; step += 2) {
                uint q0 = best_block.x & 0xFFFFu;
                uint q1 = best_block.x >> 16;
                const uint q = c < 3u ? q0 : q1;
                const int value = int((q >> shift) & top) + step;
                if (value < 0 || value > int(top)) {
                    continue;
                }
                const uint moved = (q & ~(top << shift)) | (uint(value) << shift);
                if (c < 3u) {
                    q0 = moved;
                } else {
                    q1 = moved;
                }
                uvec2 block;
                const float error = FitFourColor(real, q0, q1, block);
                if (error < best) {
                    best = error;
                    best_block = block;
                    improved = true;
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return best_block;
}

// BC4 block (and BC3's alpha block) of `values`: eight-value mode (value0 > value1), indices
// 0 = value0, 1 = value1, 2..7 the ramp from value0 to value1.
uvec2 EncodeChannel(in float values[16]) {
    float lo = 1.0;
    float hi = 0.0;
    for (int i = 0; i < 16; ++i) {
        lo = min(lo, values[i]);
        hi = max(hi, values[i]);
    }
    const uint a0 = uint(round(hi * 255.0));
    const uint a1 = uint(round(lo * 255.0));
    uint bits_lo = a0 | (a1 << 8);
    uint bits_hi = 0u;
    if (a0 == a1) {
        return uvec2(bits_lo, bits_hi);
    }
    for (uint i = 0u; i < 16u; ++i) {
        const float t = (values[i] * 255.0 - float(a1)) / float(a0 - a1);
        const uint q = uint(clamp(round(t * 7.0), 0.0, 7.0));
        const uint index = q == 7u ? 0u : (q == 0u ? 1u : 8u - q);
        const uint bit = 16u + 3u * i;
        if (bit < 32u) {
            bits_lo |= index << bit;
            if (bit > 29u) {
                bits_hi |= index >> (32u - bit);
            }
        } else {
            bits_hi |= index << (bit - 32u);
        }
    }
    return uvec2(bits_lo, bits_hi);
}

uvec2 EncodeAlpha() {
    float values[16];
    for (int i = 0; i < 16; ++i) {
        values[i] = Texel(i).a;
    }
    return EncodeChannel(values);
}

uvec2 EncodeRed() {
    float values[16];
    for (int i = 0; i < 16; ++i) {
        values[i] = Texel(i).r;
    }
    return EncodeChannel(values);
}
