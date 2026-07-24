//===- OpenMPKernelVersioning.cpp - OpenMP kernel versioning --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is intended to version OpenMP device kernels when profitable.
// TODO: explain file
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPKernelVersioning.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"

using namespace llvm;

static cl::opt<bool> DisableOpenMPKernelVersioning(
    "openmp-kernel-versioning-disable",
    cl::desc("Disable OpenMP kernel versioning."), cl::Hidden,
    cl::init(false));

namespace {

class OpenMPKernelVersioning {
public:
  explicit OpenMPKernelVersioning(Module &M)
      : M(M), Kernels(omp::getDeviceKernels(M)) {}

  bool run() {
    if (!omp::isOpenMPDevice(M) || Kernels.empty())
      return false;

    // SU3 specific optimizations.
    return versionKernel(*Kernels[0]);
  }

private:
  static CallBase *findCallByCalleeName(Function &F, StringRef CalleeName) {
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (Function *Callee = CB->getCalledFunction())
            if (Callee->getName() == CalleeName)
              return CB;

    return nullptr;
  }

  bool versionKernel(Function &Kernel) {
    #define THREADS_PER_TEAM 128
    /* #define LATTICE_DIM 50 */
    #define LATTICE_DIM 32
    Constant *NumWorkItems = ConstantInt::get(Kernel.getArg(0)->getType(), LATTICE_DIM*LATTICE_DIM*LATTICE_DIM*LATTICE_DIM*THREADS_PER_TEAM);
    Constant *TotalSites = ConstantInt::get(Kernel.getArg(1)->getType(), LATTICE_DIM*LATTICE_DIM*LATTICE_DIM*LATTICE_DIM);

    std::string OutlinedName = (Kernel.getName() + "_omp_outlined").str();
    CallBase *OutlinedCB = findCallByCalleeName(Kernel, OutlinedName);
    assert(OutlinedCB && "Expected a call to the first outlined function");
    Function *OutlinedFn = OutlinedCB->getCalledFunction();
    assert(OutlinedFn && "Expected a direct call to the first outlined function");
    CallBase *Parallel60CB = findCallByCalleeName(*OutlinedFn, "__kmpc_parallel_60");
    assert(Parallel60CB && "Expected a call to __kmpc_parallel_60");
    Function *ParallelRegion = cast<Function>(
        Parallel60CB->getArgOperand(5)->stripPointerCasts());

    Kernel.getArg(0)->replaceAllUsesWith(NumWorkItems);
    Kernel.getArg(1)->replaceAllUsesWith(TotalSites);
    OutlinedFn->getArg(2)->replaceAllUsesWith(NumWorkItems);
    OutlinedFn->getArg(3)->replaceAllUsesWith(TotalSites);
    ParallelRegion->getArg(4)->replaceAllUsesWith(NumWorkItems);
    ParallelRegion->getArg(5)->replaceAllUsesWith(TotalSites);

    return true;
  }

  Module &M;
  omp::KernelSet Kernels;
};

} // end anonymous namespace

PreservedAnalyses OpenMPKernelVersioningPass::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  if (DisableOpenMPKernelVersioning)
    return PreservedAnalyses::all();

  OpenMPKernelVersioning OKV(M);
  if (!OKV.run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
