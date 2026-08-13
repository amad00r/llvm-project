//===- LogicalSignatureAnalysis.h - Logical signature analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H
#define LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class LogicalSignatureAnalysis
    : public AnalysisInfoMixin<LogicalSignatureAnalysis> {
  friend AnalysisInfoMixin<LogicalSignatureAnalysis>;
  LLVM_ABI static AnalysisKey Key;

public:
  struct EstimateLogicalSignature {};

  LLVM_ABI EstimateLogicalSignature run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_LOGICALSIGNATUREANALYSIS_H
