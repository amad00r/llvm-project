//===- IPO/OpenMPKernelVersioning.h - OpenMP kernel versioning --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMPKERNELVERSIONING_H
#define LLVM_TRANSFORMS_IPO_OPENMPKERNELVERSIONING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// OpenMP GPU kernel versioning pass.
class OpenMPKernelVersioningPass
    : public OptionalPassInfoMixin<OpenMPKernelVersioningPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMPKERNELVERSIONING_H
