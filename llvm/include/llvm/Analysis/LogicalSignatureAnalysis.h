//===- LogicalSignatureAnalysis.h - Logical signature analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H
#define LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"
#include <variant>

namespace llvm {

class LogicalSignatureAnalysis
    : public AnalysisInfoMixin<LogicalSignatureAnalysis> {
  friend AnalysisInfoMixin<LogicalSignatureAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  class ExtendedType {
  public:
    using VariantTy = std::variant<
      std::reference_wrapper<const ExtendedType>, 
      std::reference_wrapper<const Type>>;
    VariantTy Variant;
    LLVM_ABI void print(raw_ostream &) const;
  };

  class EstimatedLogicalSignature {
    friend LogicalSignatureAnalysis;
  public:
    SmallVector<std::reference_wrapper<const ExtendedType>, 8> ArgTypes;
  private:
    SpecificBumpPtrAllocator<ExtendedType> Allocator;
  };

  using Result = EstimatedLogicalSignature;

  LLVM_ABI Result run(Function &, FunctionAnalysisManager &);
};

class LogicalSignaturePrinterPass
    : public RequiredPassInfoMixin<LogicalSignaturePrinterPass> {
  raw_ostream &OS;

public:
  explicit LogicalSignaturePrinterPass(raw_ostream &OS) : OS(OS) {}

  LLVM_ABI PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H
