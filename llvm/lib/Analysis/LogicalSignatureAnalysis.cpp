//===- LogicalSignatureAnalysis.cpp - Logical signature analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LogicalSignatureAnalysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "logical-signature"

AnalysisKey LogicalSignatureAnalysis::Key;

namespace {

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

struct Algorithm {
  using ExtendedType = LogicalSignatureAnalysis::ExtendedType;
  struct InProgress {};
  using TypeOrInProgress = std::variant
    < InProgress
    , std::reference_wrapper<const ExtendedType>
    >;

  SpecificBumpPtrAllocator<ExtendedType> &Allocator;
  SmallDenseMap<const Value *, TypeOrInProgress, 16> ValueTypeMap;

  Algorithm(SpecificBumpPtrAllocator<ExtendedType> &A) : Allocator(A) {}

  const ExtendedType &getImplementationType(const Value &V) {
    return *new (Allocator.Allocate()) ExtendedType{ ExtendedType::VariantTy(*V.getType()) };
  }

  TypeOrInProgress wrap(const TypeOrInProgress &Ty) {
    return std::visit(overloaded{
      [](InProgress) -> TypeOrInProgress { return InProgress{}; },
      [&](const ExtendedType &Ty) -> TypeOrInProgress { return *new (Allocator.Allocate()) ExtendedType{ ExtendedType::VariantTy(Ty) }; }
    }, Ty);
  }

  static const ExtendedType *unwrap(const ExtendedType &Ty) {
    return std::visit(overloaded{
      [](const Type &) -> const ExtendedType * { return nullptr; },
      [](const ExtendedType &Inner) -> const ExtendedType * { return &Inner; }
    }, Ty.Variant);
  }

  static const Type *getType(const ExtendedType &Ty) {
    return std::visit(overloaded{
      [](const Type &Inner) -> const Type * { return &Inner; },
      [](const ExtendedType &) -> const Type * { return nullptr; }
    }, Ty.Variant);
  }

  static bool isStructurallyEqual(const ExtendedType &ET1, const ExtendedType &ET2) {
    const Type *T1 = getType(ET1);
    const Type *T2 = getType(ET2);
    if (T1 != T2) return false;
    if (T1) return true;
    const ExtendedType *UnwrapET1 = unwrap(ET1);
    const ExtendedType *UnwrapET2 = unwrap(ET2);
    assert(UnwrapET1);
    assert(UnwrapET2);
    return isStructurallyEqual(*UnwrapET1, *UnwrapET2);
  }

  template <typename ...Args>
  static void debug(const Value &V, const ExtendedType &Ty, Args &&... args) {
    LLVM_DEBUG(
      V.printAsOperand(dbgs(), false);
      dbgs() << "<";
      Ty.print(dbgs());
      dbgs() << "> (";
      ((dbgs() << std::forward<Args>(args)), ...);
      dbgs() << ")\n");
  }

  TypeOrInProgress estimateUsedType(const Use &U) {
    const Value &V = *U.get();
    const Instruction &I = *cast<Instruction>(U.getUser());
    switch (I.getOpcode()) {
    case Instruction::Load:
      return wrap(estimateType(I));
    case Instruction::Store: {
      const StoreInst &SI = cast<StoreInst>(I);
      if (&V == SI.getPointerOperand())
        return wrap(estimateType(*SI.getValueOperand()));
      assert(&V == SI.getValueOperand());
      return std::visit(overloaded{
        [](InProgress) -> TypeOrInProgress { return InProgress{}; },
        [&](const ExtendedType &Ty) -> TypeOrInProgress {
          return std::visit(overloaded{
            [&](const Type &Inner) -> const ExtendedType & {
              assert(Inner.isPointerTy());
              return getImplementationType(V);
            },
            [](const ExtendedType &Inner) -> const ExtendedType & { return Inner; }
          }, Ty.Variant);
        }
      }, estimateType(*SI.getPointerOperand()));
    }
    case Instruction::Call: {
      const Function *F = cast<CallInst>(I).getCalledFunction();
      if (!F)
        return getImplementationType(V);
      return estimateType(*F->getArg(U.getOperandNo()));
    }
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
    case Instruction::Trunc:
      return estimateType(I);
    default:
      return getImplementationType(V);
    }
  }

  TypeOrInProgress estimateType(const Value &V) {
    const auto It = ValueTypeMap.find(&V);
    if (It != ValueTypeMap.end())
      return It->second;

    if (V.use_empty()) {
      const ExtendedType &ResultTy = getImplementationType(V);
      ValueTypeMap[&V] = ResultTy;
      debug(V, ResultTy, "fallback reason: unused");
      return ResultTy;
    }

    ValueTypeMap[&V] = InProgress{};

    // The logical type of a value is its implementation type by default
    SmallVector<std::reference_wrapper<const ExtendedType>, 8> UsedTypes;

    // Try to refine the logical type based on the uses of the value
    for (const Use &U : V.uses()) {
      if (isa<Instruction>(U.getUser())) {
        std::visit(overloaded{
          [](InProgress) {},
          [&](const ExtendedType &Ty) {
            UsedTypes.emplace_back(Ty);
            debug(V, Ty, "used in ", *U.getUser());
          }
        }, estimateUsedType(U));
      } else {
        const ExtendedType &ExtTy = getImplementationType(V);
        ValueTypeMap[&V] = ExtTy;
        UsedTypes.emplace_back(ExtTy);
        debug(V, ExtTy, "used in a non-instruction ", *U.getUser());
      }
    }

    if (UsedTypes.empty()) {
      const ExtendedType &ResultTy = getImplementationType(V);
      ValueTypeMap[&V] = ResultTy;
      debug(V, ResultTy, "fallback reason: no available used types");
      return ResultTy;
    }

    for (size_t I = 1; I < UsedTypes.size(); ++I) {
      if (!isStructurallyEqual(UsedTypes[0], UsedTypes[I])) {
        const ExtendedType &ResultTy = getImplementationType(V);
        ValueTypeMap[&V] = ResultTy;
        debug(V, ResultTy, "fallback reason: conservative");
        return ResultTy;
      }
    }

    const ExtendedType &ResultTy = UsedTypes[0].get();
    ValueTypeMap[&V] = ResultTy;
    debug(V, ResultTy, "deduced from used types");
    return ResultTy;
  }

  const ExtendedType &operator()(const Argument &Arg) {
    return std::visit(overloaded{
      [&](InProgress) -> const ExtendedType & { return getImplementationType(Arg); },
      [](const ExtendedType &Ty) -> const ExtendedType & { return Ty; }
    }, estimateType(Arg));
  }
};

} // namespace

void LogicalSignatureAnalysis::ExtendedType::print(raw_ostream &OS) const {
  std::visit(overloaded{
    [&](const ExtendedType &Ty) { Ty.print(OS); OS << "*"; },
    [&](const Type &Ty) { Ty.print(OS); }
  }, Variant);
}

LogicalSignatureAnalysis::EstimatedLogicalSignature
LogicalSignatureAnalysis::run(Function &F, FunctionAnalysisManager &) {
  LLVM_DEBUG(dbgs() << F << "\n");
  EstimatedLogicalSignature ELS;
  Algorithm Algo(ELS.Allocator);
  for (const Argument &Arg : F.args())
    ELS.ArgTypes.emplace_back(Algo(Arg));
  return ELS;
}

PreservedAnalyses
LogicalSignaturePrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  const auto &EstimatedLogicalSignature = AM.getResult<LogicalSignatureAnalysis>(F);
  OS << "LogicalSignatureAnalysis estimates " << F.getName() << "(";
  if (const auto NumArgs = EstimatedLogicalSignature.ArgTypes.size()) {
    EstimatedLogicalSignature.ArgTypes[0].get().print(OS);
    for (size_t I = 1; I < NumArgs; ++I) {
      OS << ", ";
      EstimatedLogicalSignature.ArgTypes[I].get().print(OS);
    }
  }
  OS << ")\n";
  return PreservedAnalyses::all();
}
