//===- KernelArgInfo.h - Kernel argument info -----------------*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H
#define LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H

#include "llvm/Support/Endian.h"
#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>

namespace llvm {

struct KernelArgInfo {
public:
  using EncodeType = uint32_t;

private:
  using PayloadType = uint16_t;
  using IndirectionType = uint8_t;

  static constexpr unsigned PayloadShift = 8;
  static constexpr unsigned PayloadBits = 16;
  static constexpr unsigned IndirectionShift = PayloadShift + PayloadBits;
  static constexpr EncodeType KindMask =
      static_cast<EncodeType>((1u << PayloadShift) - 1);
  static constexpr EncodeType PayloadMask =
      static_cast<EncodeType>((1u << PayloadBits) - 1);
  static constexpr EncodeType IndirectionMask = KindMask;

  EncodeType Encoded;

public:
  enum class Kind : uint8_t {
    Integer = 0,
    Float = 1,
    Double = 2,
    Unknown = KindMask,
  };

private:
  explicit KernelArgInfo(Kind K, PayloadType Payload,
                         IndirectionType Indirection)
      : Encoded((static_cast<EncodeType>(Indirection) << IndirectionShift) |
                (static_cast<EncodeType>(Payload) << PayloadShift) |
                static_cast<EncodeType>(K)) {}

  explicit KernelArgInfo(EncodeType Encoded) : Encoded(Encoded) {}

  PayloadType getPayload() const {
    return (Encoded >> PayloadShift) & PayloadMask;
  }

public:
  static KernelArgInfo getIntegerTy(PayloadType BitWidth,
                                    IndirectionType Indirection = 0) {
    return KernelArgInfo(Kind::Integer, BitWidth, Indirection);
  }

  static KernelArgInfo getFloatTy(IndirectionType Indirection = 0) {
    return KernelArgInfo(Kind::Float, 0, Indirection);
  }

  static KernelArgInfo getDoubleTy(IndirectionType Indirection = 0) {
    return KernelArgInfo(Kind::Double, 0, Indirection);
  }

  static KernelArgInfo getPointerTy(KernelArgInfo Pointee) {
    return KernelArgInfo(Pointee.getKind(), Pointee.getPayload(),
                         Pointee.getIndirection() + 1);
  }

  static KernelArgInfo getOpaquePointerTy(IndirectionType Indirection = 0) {
    return KernelArgInfo(Kind::Unknown, 0, Indirection + 1);
  }

  static KernelArgInfo getUnknownTy() {
    return KernelArgInfo(Kind::Unknown, 0, 0);
  }

  Kind getKind() const { return static_cast<Kind>(Encoded & KindMask); }

  PayloadType getIntegerBitWidth() const { return getPayload(); }

  IndirectionType getIndirection() const {
    return (Encoded >> IndirectionShift) & IndirectionMask;
  }

  EncodeType getEncodedLE() const {
    EncodeType E;
    support::endian::write32le(&E, Encoded);
    return E;
  }

  static KernelArgInfo fromEncodedLE(const void *P) {
    return KernelArgInfo(support::endian::read32le(P));
  }

  std::string typeStr() const {
    switch (getKind()) {
    case Kind::Integer:
      return "i" + std::to_string(getIntegerBitWidth()) +
             std::string(getIndirection(), '*');
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

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_KERNELARGINFO_H
