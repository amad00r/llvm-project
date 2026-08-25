//===- MaterializedKernelInfo.h - Materialized kernel info ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures used to encode materialized kernel
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
#define LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringExtras.h"
#include <variant>
#include <string_view>

namespace llvm {

struct KernelInfo {
  struct Argument {
    class Type {
    private:
      template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
      template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

    public:
      struct Unknown {};
      struct Int { unsigned BitWidth; };
      struct Float {};
      struct Double {};
      struct Ptr {};
      std::variant<Unknown, Int, Float, Double, Ptr> Variant;

      template <typename ...Fn>
      decltype(auto) visit(Fn &&... F) const {
        return std::visit(overloaded{ std::forward<Fn>(F)... }, Variant);
      }

      SmallString<16> str() const {
        using namespace std::string_view_literals;
        return visit(
          [](Unknown) -> SmallString<16> { return { "unknown"sv }; },
          [](Int I) -> SmallString<16> { return { "i"sv, std::to_string(I.BitWidth) }; },
          [](Float) -> SmallString<16> { return { "float"sv }; },
          [](Double) -> SmallString<16> { return { "double"sv }; },
          [](Ptr) -> SmallString<16> { return { "ptr"sv }; }
        );
      }
    };

    Type Ty;
  
    Argument(Type Ty_) : Ty(Ty_) {}

    Argument(StringRef Str)
      : Ty(StringSwitch<Type>(Str)
        .Case("unknown", { Type::Unknown{} })
        .Case("float", { Type::Float{} })
        .Case("double", { Type::Double{} })
        .Case("ptr", { Type::Ptr{} })
        .Default([&]() -> Type {
          unsigned BitWidth;
          if (Str.consume_front("i") && !Str.getAsInteger(10, BitWidth))
            return { Type::Int{ BitWidth } };
          return { Type::Unknown{} };
        }())
      )
    {
    }
  };

  SmallVector<Argument, 8> Args;

  KernelInfo() = default;

  KernelInfo(StringRef Str)
    : Args([&Str]() -> decltype(Args) {
      if (Str.empty())
        return {};
      decltype(Args) A;
      for (const StringRef EncodedArgType : split(Str, ';'))
        A.emplace_back(EncodedArgType);
      return A;
    }())
  {
  }

  SmallString<128> str() const {
    if (Args.empty())
      return {};
    SmallString<128> KernelInfoText(Args[0].Ty.str());
    for (size_t I = 1; I < Args.size(); ++I)
      KernelInfoText.append({ ";", Args[I].Ty.str() });
    return KernelInfoText;
  }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
