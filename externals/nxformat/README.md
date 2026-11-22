# nxformat reader

Header-only reader for `.nx` archives, from [nxformat](https://github.com/fab918/nxformat)
by fab918 (MIT; see `LICENSE`). These headers are copied unchanged from Lichii PR
[#1887](https://github.com/zortos293/Lichii/pull/1887) (nxformat 1.0.4 with its
relative pointer arithmetic fixes).

The emulator integration ports Lichii PRs [#923](https://github.com/zortos293/Lichii/pull/923),
[#1078](https://github.com/zortos293/Lichii/pull/1078),
[#1232](https://github.com/zortos293/Lichii/pull/1232) and
[#1887](https://github.com/zortos293/Lichii/pull/1887).

Loading archives uses these vendored headers and the emulator's Zstandard library;
it requires no nxformat download. The Qt conversion tools separately use nxformat
1.0.4's `static_nozstd` library, downloaded with pinned SHA-256 digests by
`CMakeModules/NxFormat.cmake`. Set `EGLANTINE_NXFORMAT_ROOT` to an unpacked release
to use a local copy. When the download is unavailable, loading still works and only
the conversion tools are disabled. No nxformat shared library needs to be packaged.
