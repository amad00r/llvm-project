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

  static std::string getGlobalNameFor(StringRef KernelName) {
    return (KernelName + "_kernel_info").str();
  }

  struct Argument {
    struct Type {
      struct Unknown {
        bool operator==(Unknown) const { return true; }
        bool operator!=(Unknown) const { return false; }
      };
      struct Int {
        unsigned BitWidth;
        bool operator==(const Int &Other) const { return BitWidth == Other.BitWidth; }
        bool operator!=(const Int &Other) const { return !(*this == Other); }
      };
      struct Float {
        bool operator==(Float) const { return true; }
        bool operator!=(Float) const { return false; }
      };
      struct Double {
        bool operator==(Double) const { return true; }
        bool operator!=(Double) const { return false; }
      };
      struct Ptr {
        bool operator==(Ptr) const { return true; }
        bool operator!=(Ptr) const { return false; }
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
      struct NoLoop {};
      struct NoAlias {};
      struct ArgAlignment { unsigned ArgNo, Alignment; };
      std::variant<NoLoop, NoAlias, ArgAlignment> Variant;
      template <typename ...Fn>
      decltype(auto) visit(Fn &&... F) const {
        return std::visit(overloaded{ std::forward<Fn>(F)... }, Variant);
      }
    };
    
    // TODO: this should be a set if we want to compare Versions. otherwise, versions with permutated specializations are considered to be different
    SmallVector<Specialization, 8> Specializations;

    Version() = default;

    Version(std::initializer_list<Specialization> Init)
      : Specializations(Init)
    {}

    Version(StringRef Str) {
      const bool Ok = Str.consume_front(".");
      assert(Ok && "A version always starts with a dot");
      for (StringRef Specialization : split(Str, '.')) {
        if (Specialization == StringRef("noloop")) {
          Specializations.push_back({ Specialization::NoLoop{} });
          continue;
        }
        if (Specialization == StringRef("noalias")) {
          Specializations.push_back({ Specialization::NoAlias{} });
          continue;
        }
        if (Specialization.consume_front("align")) {
          const auto [AlignStr, ArgNoStr] = Specialization.split('$');
          unsigned Align, ArgNo;
          const bool AlignOk = AlignStr.getAsInteger(10, Align);
          assert(!AlignOk);
          const bool ArgNoOk = ArgNoStr.getAsInteger(10, ArgNo);
          assert(!ArgNoOk);
          Specializations.push_back({ Specialization::ArgAlignment{ ArgNo, Align } });
          continue;
        }
        llvm_unreachable("unexpected specialization");
      }
    }

    SmallString<128> str() const {
      SmallString<128> VersionText;
      raw_svector_ostream OS(VersionText);
      for (const auto &S : Specializations) S.visit(
        [&](const Specialization::NoLoop &) { OS << ".noloop"; },
        [&](const Specialization::NoAlias &) { OS << ".noalias"; },
        [&](const Specialization::ArgAlignment &AA) { OS << ".align" << AA.Alignment << '$' << AA.ArgNo; }
      );
      return VersionText;
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
