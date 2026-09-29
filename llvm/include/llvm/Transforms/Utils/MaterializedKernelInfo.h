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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include <variant>
#include <string>
#include <utility>

namespace llvm {

struct KernelInfo {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

  static SmallString<128> getGlobalNameFor(StringRef KernelName) {
    return { KernelName, "_kernel_info" };
  }

  struct Argument {
    struct Type {
      struct Unknown {
        bool operator==(Unknown) const { return true; }
      };
      struct Int {
        unsigned BitWidth;
        bool operator==(const Int &Other) const { return BitWidth == Other.BitWidth; }
      };
      struct Float {
        bool operator==(Float) const { return true; }
      };
      struct Double {
        bool operator==(Double) const { return true; }
      };
      struct Ptr {
        bool operator==(Ptr) const { return true; }
      };
      std::variant<Unknown, Int, Float, Double, Ptr> Variant;

      template <typename ...Fn>
      decltype(auto) visit(Fn &&... F) const {
        return std::visit(overloaded{ std::forward<Fn>(F)... }, Variant);
      }

      bool operator==(const Type &Other) const {
        return Variant == Other.Variant;
      }
      bool operator!=(const Type &Other) const { return !(*this == Other); }

      SmallString<16> str() const {
        return visit(
          [](Unknown) -> SmallString<16> { return { "unknown" }; },
          [](Int I) -> SmallString<16> { return { "i", std::to_string(I.BitWidth) }; },
          [](Float) -> SmallString<16> { return { "float" }; },
          [](Double) -> SmallString<16> { return { "double" }; },
          [](Ptr) -> SmallString<16> { return { "ptr" }; }
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

    bool operator==(const Argument &Other) const { return Ty == Other.Ty; }
    bool operator!=(const Argument &Other) const { return !(*this == Other); }
  };

  struct Version {
    struct Specialization {
      struct NoLoop {
        bool operator==(NoLoop) const { return true; }
      };
      struct NoAlias {
        bool operator==(NoAlias) const { return true; }
      };
      struct Align16 {
        bool operator==(Align16) const { return true; }
      };
      struct Unroll2 {
        bool operator==(Unroll2) const { return true; }
      };
      std::variant<NoLoop, NoAlias, Align16, Unroll2> Variant;
      template <typename ...Fn>
      decltype(auto) visit(Fn &&... F) const {
        return std::visit(overloaded{ std::forward<Fn>(F)... }, Variant);
      }
      bool operator==(const Specialization &Other) const {
          return Variant == Other.Variant;
        }
      bool operator!=(const Specialization &Other) const { return !(*this == Other); }
      Specialization(StringRef Str)
        : Variant([&]() -> decltype(Variant) {
          if (Str == "noloop") return { NoLoop{} };
          if (Str == "noalias") return { NoAlias{} };
          if (Str == "align16") return { Align16{} };
          if (Str == "unroll2") return { Unroll2{} };
          llvm_unreachable("unexpected specialization");
        }())
      {}
      SmallString<16> str() const {
        return visit(
          [](NoLoop) -> SmallString<16> { return { ".noloop" }; },
          [](NoAlias) -> SmallString<16> { return { ".noalias" }; },
          [](Align16) -> SmallString<16> { return { ".align16" }; },
          [](Unroll2) -> SmallString<16> { return { ".unroll2" }; }
        );
      }
    };

    struct SpecializationSequence {
      SmallVector<Specialization, 8> Specializations;

      SpecializationSequence() = default;

      SpecializationSequence(StringRef Str) {
        const bool Ok = Str.consume_front(".");
        assert(Ok && "A version always starts with a dot");
        for (StringRef SpecializationStr : split(Str, '.'))
          Specializations.emplace_back(SpecializationStr);
      }

      SmallString<64> str() const {
        SmallString<64> Text;
        raw_svector_ostream OS(Text);
        for (const auto &S : Specializations)
          OS << S.str();
        return Text;
      }
    };

    SpecializationSequence Specializations;
    std::string Symbol;

    Version() = default;

    Version(StringRef Str) {
      assert(Str.count('@') == 1);
      std::tie(Specializations, Symbol) = Str.split('@');
    }

    Version(SpecializationSequence Specializations, StringRef Symbol)
      : Specializations(Specializations), Symbol(Symbol)
    {
      assert(!Specializations.Specializations.empty());
      assert(!Symbol.empty());
    }

    SmallString<128> str() const {
      return { Specializations.str(), "@", Symbol };
    }
  };

  std::optional<SmallVector<Argument, 8>> Args;
  std::optional<SmallVector<Version, 8>> Versions;

  KernelInfo() = default;

  static constexpr StringLiteral SectionSeparator = " ";
  static constexpr StringLiteral SectionNameSeparator = ":";
  static constexpr StringLiteral ElementSeparator = ";";
  static constexpr StringLiteral ArgSection = "args";
  static constexpr StringLiteral VersionSection = "versions";

  KernelInfo(StringRef Str) {
    if (Str.empty())
      return;

    for (const StringRef Section : split(Str, SectionSeparator)) {
      const auto [SectionName, SectionContent] = Section.split(SectionNameSeparator);
      if (SectionName == ArgSection) {
        auto &A = Args.emplace();
        if (!SectionContent.empty())
          for (const StringRef EncodedArgType : split(SectionContent, ElementSeparator))
            A.emplace_back(EncodedArgType);
        continue;
      }
      if (SectionName == VersionSection) {
        auto &V = Versions.emplace();
        if (!SectionContent.empty())
          for (const StringRef EncodedVersion : split(SectionContent, ElementSeparator))
            V.emplace_back(EncodedVersion);
        continue;
      }
    }
  }

  SmallString<128> str() const {
    SmallString<128> KernelInfoText;
    raw_svector_ostream OS(KernelInfoText);

    if (Args) {
      OS << ArgSection << SectionNameSeparator;
      interleave(*Args, OS, [&](const Argument &Arg) { OS << Arg.Ty.str(); }, ElementSeparator);
    }

    if (Versions) {
      if (!KernelInfoText.empty())
        OS << SectionSeparator;
      OS << VersionSection << SectionNameSeparator;
      interleave(*Versions, OS, [&](const Version &V) { OS << V.str(); }, ElementSeparator);
    }

    return KernelInfoText;
  }
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MATERIALIZEDKERNELINFO_H
