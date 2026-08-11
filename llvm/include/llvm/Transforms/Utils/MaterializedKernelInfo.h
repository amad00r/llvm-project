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
#include <sstream>

namespace llvm {

struct KernelArgTypeInfo {
private:
  static constexpr unsigned PayloadShift = 8;
  static constexpr unsigned PayloadBits = 16;
  static constexpr unsigned IndirectionShift = PayloadShift + PayloadBits;
  static constexpr uint32_t KindMask =
      static_cast<uint32_t>((1u << PayloadShift) - 1);
  static constexpr uint32_t PayloadMask =
      static_cast<uint32_t>((1u << PayloadBits) - 1);

public:
  enum class Kind : uint8_t {
    Integer = 0,
    Float = 1,
    Double = 2,
    Unknown = KindMask,
  };

private:
  uint32_t Encoded;

  constexpr KernelArgTypeInfo(Kind K, uint16_t Payload, uint8_t Indirection)
      : Encoded((static_cast<uint32_t>(Indirection) << IndirectionShift) |
                (static_cast<uint32_t>(Payload) << PayloadShift) |
                static_cast<uint32_t>(K)) {}

  constexpr uint16_t getPayload() const {
    return (Encoded >> PayloadShift) & PayloadMask;
  }

public:
  static constexpr KernelArgTypeInfo getIntegerTy(uint32_t BitWidth,
                                                  uint32_t Indirection = 0) {
    return KernelArgTypeInfo(Kind::Integer, BitWidth, Indirection);
  }

  static constexpr KernelArgTypeInfo getFloatTy(uint32_t Indirection = 0) {
    return KernelArgTypeInfo(Kind::Float, 0, Indirection);
  }

  static constexpr KernelArgTypeInfo getDoubleTy(uint32_t Indirection = 0) {
    return KernelArgTypeInfo(Kind::Double, 0, Indirection);
  }

  static constexpr KernelArgTypeInfo
  getPointerTy(KernelArgTypeInfo PointeeTy) {
    return KernelArgTypeInfo(PointeeTy.getKind(), PointeeTy.getPayload(),
                             PointeeTy.getIndirection() + 1);
  }

  static constexpr KernelArgTypeInfo
  getOpaquePointerTy(uint32_t Indirection = 0) {
    return KernelArgTypeInfo(Kind::Unknown, 0, Indirection + 1);
  }

  static constexpr KernelArgTypeInfo getPointerTy() {
    return getOpaquePointerTy();
  }

  static constexpr KernelArgTypeInfo getUnknownTy() {
    return KernelArgTypeInfo(Kind::Unknown, 0, 0);
  }

  constexpr Kind getKind() const {
    return static_cast<Kind>(Encoded & KindMask);
  }

  constexpr uint32_t getIntegerBitWidth() const {
    return getPayload();
  }

  constexpr uint32_t getIndirection() const {
    return (Encoded >> IndirectionShift) & KindMask;
  }

  constexpr uint32_t getEncoded() const { return Encoded; }

  std::string typeStr() const {
    switch (getKind()) {
    case Kind::Integer:
      return "i" + std::to_string(getIntegerBitWidth()) + std::string(getIndirection(), '*');
    case Kind::Float:
      return "float" + std::string(getIndirection(), '*');
    case Kind::Double:
      return "double" + std::string(getIndirection(), '*');
    case Kind::Unknown:
      if (getIndirection() == 0)
        return "unknown";
      return "ptr" + std::string(getIndirection() - 1, '*');
    }
  }

  std::string valueStr(void *Value) const {
    assert(Value);
    if (getIndirection() > 0) {
      std::ostringstream oss;
      oss << *reinterpret_cast<void **>(Value);
      return oss.str();
    }

    switch (getKind()) {
    case Kind::Integer:
      switch (getIntegerBitWidth()) {
      case 8:
        return std::to_string(*reinterpret_cast<uint8_t *>(Value));
      case 16:
        return std::to_string(*reinterpret_cast<uint16_t *>(Value));
      case 32:
        return std::to_string(*reinterpret_cast<uint32_t *>(Value));
      case 64:
        return std::to_string(*reinterpret_cast<uint64_t *>(Value));
      default:
        return "<unsupported bit width>";
      }
    case Kind::Float:
      return std::to_string(*reinterpret_cast<float *>(Value));
    case Kind::Double:
      return std::to_string(*reinterpret_cast<double *>(Value));
    case Kind::Unknown:
      return "<no representation>";
    }
  }
};

static_assert(sizeof(KernelArgTypeInfo) == sizeof(uint32_t));

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
