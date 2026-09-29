//===- MaterializeKernelInfo.h - Materialize kernel info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MaterializeKernelInfoPass, a module transform that
// materializes compile-time kernel information so it is available at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MATERIALIZEKERNELINFO_H
#define LLVM_TRANSFORMS_UTILS_MATERIALIZEKERNELINFO_H

#include "llvm/Transforms/Utils/MaterializedKernelInfo.h"
#include "llvm/IR/PassManager.h"
#include <optional>

namespace llvm {

namespace kernel_info_utils {

LLVM_ABI std::optional<KernelInfo> parseGlobalFor(Function &);
LLVM_ABI void createGlobalFor(Function &, const KernelInfo &);
LLVM_ABI void createOrReplaceGlobalFor(Function &, const KernelInfo &);

} // end namespace kernel_info_utils

class MaterializeKernelInfoPass
    : public OptionalPassInfoMixin<MaterializeKernelInfoPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MATERIALIZEKERNELINFO_H
