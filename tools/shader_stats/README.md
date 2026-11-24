# shader_stats — read a shader's register and code-size numbers offline

A compute shader's cost on AMD hardware is set by things the game log can't tell
you: how many VGPRs the register allocator needed, how many waves that leaves
resident per SIMD, and how much code the instruction cache has to stream.
`RADV_DEBUG=shaderstats` prints all of it, but only when the driver actually
compiles a pipeline — which means booting a title and hoping its pipeline cache
is cold.

`vkcompile` sidesteps that: it creates a compute pipeline from one `.spv` and
destroys it. Nothing is dispatched, no game runs, and a measurement takes about
a second instead of a boot.

```
nix shell nixpkgs#gcc nixpkgs#vulkan-headers nixpkgs#vulkan-loader -c \
  gcc -I$VULKAN_HEADERS/include -L$VULKAN_LOADER/lib -o vkcompile vkcompile.c -lvulkan

glslangValidator -V --target-env vulkan1.2 -o astc_bc.spv astc_bc.comp
RADV_DEBUG=shaderstats ./vkcompile astc_bc.spv
```

The second argument is the binding that is a storage image (`-1` for none,
`0` or `1` to match a shader that mixes images and buffers); the third is the
push-constant range in bytes. They only need to match the module's layout well
enough for the driver's shader key.

## Reading the output

- **VGPRs** — registers per lane. Sets how many waves a SIMD holds:
  `Subgroups per SIMD` is that count, and the device's ceiling is what a trivial
  shader reports (32 on RDNA2 for wave32, verified on a Navi 23).
- **Code size** — bytes of code. RDNA has a 32 KB instruction cache per WGP
  (AMD's RDNA architecture whitepaper; Navi 24 notes say the same), so a shader
  past ~32 KB streams from L2 for every wave.
- **Spilled VGPRs / Scratch size** — non-zero means the allocator gave up and
  put live values in memory.
- **Instructions / VALU** — total work encoded, static.

## What it showed for the ASTC→BCn transcode

Measured on a Navi 23 (RX 6600, Mesa 26.3.0), `astc_bc.comp` as shipped:

| variant | VGPR | waves/SIMD | code bytes | instructions |
|---|---|---|---|---|
| fused, `mode` kind at run time (before specialisation) | 168 | 6 | 322,344 | 56,179 |
| fused, texels as `vec4 px[16]` (pre-packing) | 168 | 6 | 317,956 | 55,235 |
| **BC1, REFINE off — the pipeline the fused path runs** | **120** | **8** | **195,104** | **32,620** |
| **BC3, REFINE off** | **128** | **8** | **189,532** | **32,086** |
| BC3, REFINE on (quality knob, unused today) | 168 | 6 | 235,368 | 40,019 |
| BC4 / BC5, REFINE off (smart stand-ins) | 120 | 8 | 180,976 / 183,512 | 30,235 / 30,754 |
| two-pass decoder alone (`astc_decoder.comp`) | 168 | 6 | 102,908 | 17,996 |
| two-pass encoder alone (`bc_encoder.comp`) | 128 | 8 | 69,456 | 12,614 |

The last five rows are one module with the kind and `REFINE` as specialisation constants
(`constant_id 0` and `1`, see `astc_bc.glsl`), each row a pipeline the driver folded; pass the
values as `vkcompile <spv> -1 128 <kind> <refine>`.

Three things follow. Packing the texels into RGBA8 words changed nothing the
allocator cares about — 168 VGPRs either way — so the register bill is the
decoder's 39-word block state, not the texels. The unspecialised shader was
bigger than the two passes it replaced *plus* half again (56,179 against
30,610), with four encoder bodies and a refined fit all live at once because
`mode` chose between them at run time; at 322 KB against a 32 KB instruction
cache that is what thrashed it. And fixing the kind per pipeline gets that back:
39% less code, 42% fewer instructions, and 8 waves per SIMD instead of 6, with
the same decoder and the same arithmetic.

`spirv-opt -O` and `-Os` before compilation changed nothing measurable
(322,344 → 321,868 bytes), so this has to be won in the source, by making the
kind and `REFINE` specialization constants instead of push-constant bits.
