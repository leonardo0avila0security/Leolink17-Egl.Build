// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "core/hle/kernel/k_address_space_info.h"

using namespace Common::Literals;
using Space = Kernel::KAddressSpaceInfo;
using Type = Space::Type;

namespace {
constexpr u64 Xenoblade2Switch2 = 0x0400E95004038001ULL;
constexpr u64 Xenoblade2Switch1 = 0x0100E95004038000ULL;
} // namespace

TEST_CASE("Xenoblade 2 Switch 2 alias region holds both startup reservations", "[kernel][alias]") {
    const auto alias = Space::GetAddressSpaceSize(39, Type::Alias, Xenoblade2Switch2);
    REQUIRE(alias == 128_GiB);
    REQUIRE(alias > 2 * 60_GiB);
}

TEST_CASE("Other programs keep the default alias region", "[kernel][alias]") {
    REQUIRE(Space::GetAddressSpaceSize(39, Type::Alias) == 64_GiB);
    REQUIRE(Space::GetAddressSpaceSize(39, Type::Alias, Xenoblade2Switch1) == 64_GiB);
    REQUIRE(Space::GetAddressSpaceSize(36, Type::Alias, Xenoblade2Switch2) == 6_GiB);
    for (const auto type : {Type::Heap, Type::Stack, Type::MapSmall, Type::Map39Bit}) {
        REQUIRE(Space::GetAddressSpaceSize(39, type, Xenoblade2Switch2) ==
                Space::GetAddressSpaceSize(39, type));
    }
}

TEST_CASE("Xenoblade 2 regions fit the 39-bit code region", "[kernel][alias]") {
    // Alias, heap, kernel map and stack are placed inside the code region; with NCE it is 256 GiB.
    const auto total = Space::GetAddressSpaceSize(39, Type::Alias, Xenoblade2Switch2) +
                       Space::GetAddressSpaceSize(39, Type::Heap) +
                       Space::GetAddressSpaceSize(39, Type::MapSmall) +
                       Space::GetAddressSpaceSize(39, Type::Stack);
    REQUIRE(total < 256_GiB - 128_MiB);
    REQUIRE(total <= Space::GetAddressSpaceSize(39, Type::Map39Bit));
}
