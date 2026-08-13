//===- LogicalSignatureAnalysis.cpp - Logical signature analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LogicalSignatureAnalysis.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "logical-signature"

AnalysisKey LogicalSignatureAnalysis::Key;

LogicalSignatureAnalysis::Result
LogicalSignatureAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return Result();
}
