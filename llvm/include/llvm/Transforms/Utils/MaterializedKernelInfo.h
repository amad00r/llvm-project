//===- MaterializedKernelInfo.h - Materialized kernel info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures used to encode kernel information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
#define LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H

#include <cstdint>
#include <string>

namespace llvm {

struct KernelArgTypeInfo {
private:
  static constexpr unsigned PayloadShift = 8;
  static constexpr uint32_t KindMask =
      static_cast<uint32_t>((1u << PayloadShift) - 1);

public:
  enum class Kind : uint8_t {
    Integer = 0,
    Float = 1,
    Double = 2,
    Pointer = 3,
    Unknown = KindMask,
  };

private:
  uint32_t Encoded;

  constexpr KernelArgTypeInfo(Kind K, uint32_t Payload)
      : Encoded((Payload << PayloadShift) | static_cast<uint32_t>(K)) {}

public:
  static constexpr KernelArgTypeInfo getIntegerTy(uint32_t BitWidth) {
    return KernelArgTypeInfo(Kind::Integer, BitWidth);
  }

  static constexpr KernelArgTypeInfo getFloatTy() {
    return KernelArgTypeInfo(Kind::Float, 0);
  }

  static constexpr KernelArgTypeInfo getDoubleTy() {
    return KernelArgTypeInfo(Kind::Double, 0);
  }

  static constexpr KernelArgTypeInfo getPointerTy() {
    return KernelArgTypeInfo(Kind::Pointer, 0);
  }

  static constexpr KernelArgTypeInfo getUnknownTy() {
    return KernelArgTypeInfo(Kind::Unknown, 0);
  }

  constexpr Kind getKind() const {
    return static_cast<Kind>(Encoded & KindMask);
  }

  constexpr uint32_t getIntegerBitWidth() const {
    return Encoded >> PayloadShift;
  }

  constexpr uint32_t getEncoded() const { return Encoded; }

  std::string print() const {
    switch (getKind()) {
    case Kind::Integer:
      return "i" + std::to_string(getIntegerBitWidth());
    case Kind::Float:
      return "float";
    case Kind::Double:
      return "double";
    case Kind::Pointer:
      return "ptr";
    case Kind::Unknown:
      return "unknown";
    }
    return "unknown";
  }
};

static_assert(sizeof(KernelArgTypeInfo) == sizeof(uint32_t));

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
