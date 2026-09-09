//===- GPUKernelVersioning.h - GPU kernel versioning pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_GPUKERNELVERSIONING_H
#define LLVM_TRANSFORMS_IPO_GPUKERNELVERSIONING_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Version GPU kernels based on launch-time properties.
class GPUKernelVersioningPass
    : public OptionalPassInfoMixin<GPUKernelVersioningPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_GPUKERNELVERSIONING_H
