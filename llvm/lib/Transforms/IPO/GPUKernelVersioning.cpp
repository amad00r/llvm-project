//===- GPUKernelVersioning.cpp - GPU kernel versioning pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GPUKernelVersioning.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/SmallString.h"
#include <functional>

using namespace llvm;

#define DEBUG_TYPE "gpu-kernel-versioning"

namespace {

template <typename Fn>
CallInst *getFirstCallThatSatisfies(Function &F, const Fn &Predicate) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = llvm::dyn_cast<CallInst>(&I))
        if (std::invoke(Predicate, *CI))
          return CI;
  return nullptr;
}

CallInst *getFirstCallNamed(Function &F, StringRef Name) {
  return getFirstCallThatSatisfies(F, [Name](const CallInst &CI) {
    if (const Function *Callee = CI.getCalledFunction())
      return Callee->getName() == Name;
    return false;
  });
}

CallInst *getFirstCallThatStartsWith(Function &F, StringRef Prefix) {
  return getFirstCallThatSatisfies(F, [Prefix](const CallInst &CI) {
    if (const Function *Callee = CI.getCalledFunction())
      return Callee->getName().starts_with(Prefix);
    return false;
  });
}

CallInst *getFirstKmpcDistributeStaticInitCall(Function &F) {
  return getFirstCallThatStartsWith(F, "__kmpc_distribute_static_init_");
}
CallInst *getFirstKmpcForStaticInitCall(Function &F) {
  return getFirstCallThatStartsWith(F, "__kmpc_for_static_init_");
}
CallInst *getFirstKmpcParallel60Call(Function &F) {
  return getFirstCallNamed(F, "__kmpc_parallel_60");
}

Function *getMicrotask(const CallInst &KmpcParallel60Call) {
  return cast<Function>(KmpcParallel60Call.getArgOperand(5)->stripPointerCasts());
}

bool isDistributeParallelFor(Function &F) {
  const CallInst *Outer = getFirstKmpcDistributeStaticInitCall(F);
  if (!Outer) return false;
  const CallInst *Parallel = getFirstKmpcParallel60Call(F);
  if (!Parallel) return false;
  Function *Microtask = getMicrotask(*Parallel);
  if (!Microtask) return false;
  const CallInst *Inner = getFirstKmpcForStaticInitCall(*Microtask);
  return Inner;
}

Type &getKmpcStaticInitIntTy(const Function &F) {
  // chunk argument
  return *F.getArg(8)->getType();
}

Value &adjustOuterStaticDistribution(CallInst &I) {
  const Function *KmpcDistributeStaticInit = I.getCalledFunction();
  assert(KmpcDistributeStaticInit);

  Type &IntTy = getKmpcStaticInitIntTy(*KmpcDistributeStaticInit);

  Value &LowerBoundPtr = *I.getArgOperand(4);
  Value &UpperBoundPtr = *I.getArgOperand(5);
  Value &StridePtr = *I.getArgOperand(6);

  IRBuilder<> Builder(I.getNextNode());
  Value &UpperBound = *Builder.CreateLoad(&IntTy, &UpperBoundPtr);
  Value &LowerBound = *Builder.CreateLoad(&IntTy, &LowerBoundPtr);
  Value &TripCountMinus1 = *Builder.CreateSub(&UpperBound, &LowerBound);
  Value &TripCount = *Builder.CreateAdd(&TripCountMinus1, ConstantInt::get(&IntTy, 1));
  Builder.CreateStore(&TripCount, &StridePtr);

  if (CallInst *Fini = getFirstCallNamed(*I.getFunction(), "__kmpc_distribute_static_fini"))
    Fini->eraseFromParent();
  I.eraseFromParent();

  return TripCount;
}

void adjustInnerStaticDistribution(CallInst &I) {
  const Function *KmpcDistributeStaticInit = I.getCalledFunction();
  assert(KmpcDistributeStaticInit);
  Type &IntTy = getKmpcStaticInitIntTy(*KmpcDistributeStaticInit);
  Value &LowerBoundPtr = *I.getArgOperand(4);
  Value &UpperBoundPtr = *I.getArgOperand(5);
  Value &StridePtr = *I.getArgOperand(6);

  BasicBlock &EntryBB = I.getFunction()->getEntryBlock();
  IRBuilder<> Builder(&EntryBB, EntryBB.getFirstInsertionPt());
  FunctionType &NoArgsRetI32Ty = *FunctionType::get(Builder.getInt32Ty(), {}, false);
  const FunctionCallee OmpGetThreadNum = I.getModule()->getOrInsertFunction("omp_get_thread_num", &NoArgsRetI32Ty);
  const FunctionCallee OmpGetNumThreads = I.getModule()->getOrInsertFunction("omp_get_num_threads", &NoArgsRetI32Ty);
  const FunctionCallee OmpGetTeamNum = I.getModule()->getOrInsertFunction("omp_get_team_num", &NoArgsRetI32Ty);
  CallInst &ThreadNum = *Builder.CreateCall(OmpGetThreadNum);
  CallInst &NumThreads = *Builder.CreateCall(OmpGetNumThreads);
  CallInst &TeamNum = *Builder.CreateCall(OmpGetTeamNum);
  Value &AbsoluteThreadId = *Builder.CreateAdd(Builder.CreateMul(&TeamNum, &NumThreads), &ThreadNum);
  Value &AbsoluteThreadId64 = *Builder.CreateZExt(&AbsoluteThreadId, Builder.getInt64Ty());
  I.getFunction()->getArg(2)->replaceAllUsesWith(&AbsoluteThreadId64);
  I.getFunction()->getArg(3)->replaceAllUsesWith(&AbsoluteThreadId64);

  Builder.SetInsertPoint(I.getNextNode());
  Value *TripCount = I.getFunction()->getArg(2);
  assert(TripCount);
  Value &IsActiveThread = *Builder.CreateICmpULT(&AbsoluteThreadId, Builder.CreateTrunc(TripCount, Builder.getInt32Ty()));
  BasicBlock &OriginalBB = *Builder.GetInsertBlock();
  BasicBlock &OriginalContBB = *OriginalBB.splitBasicBlock(Builder.GetInsertPoint());
  BasicBlock &ActiveThreadBB = *BasicBlock::Create(Builder.getContext(),  "omp.active_thread.then", Builder.GetInsertBlock()->getParent(), &OriginalContBB);
  BasicBlock &InactiveThreadBB = *BasicBlock::Create(Builder.getContext(), "omp.active_thread.else", Builder.GetInsertBlock()->getParent(), &OriginalContBB);

  OriginalBB.getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(&OriginalBB);
  // TODO: Add weights. We want to improve the case where threads are active
  Builder.CreateCondBr(&IsActiveThread, &ActiveThreadBB, &InactiveThreadBB);

  Builder.SetInsertPoint(&ActiveThreadBB);
  Builder.CreateStore(&AbsoluteThreadId, &LowerBoundPtr);
  Builder.CreateStore(&AbsoluteThreadId, &UpperBoundPtr);
  Builder.CreateStore(ConstantInt::get(&IntTy, 1), &StridePtr);
  Builder.CreateBr(&OriginalContBB);

  Builder.SetInsertPoint(&InactiveThreadBB);
  Builder.CreateRetVoid();

  if (CallInst *Fini = getFirstCallNamed(*I.getFunction(), "__kmpc_for_static_fini"))
    Fini->eraseFromParent();
  I.eraseFromParent();
}

Function &specializeMicrotaskNoLoop(Function &Microtask) {
  ValueToValueMapTy VMap;
  Function &ClonedMicrotask = *CloneFunction(&Microtask, VMap);

  CallInst *ForStaticInit = getFirstKmpcForStaticInitCall(ClonedMicrotask);
  assert(ForStaticInit && "expected the microtask to contain a __kmpc_for_static_init_* call");

  adjustInnerStaticDistribution(*ForStaticInit);
  return ClonedMicrotask;
}

SmallString<64> getNoLoopSymbolName(const Function &F) {
  return { F.getName(), ".noloop" };
}

Function &specializeKernelNoLoop(Function &Kernel) {
  ValueToValueMapTy VMap;
  Function &ClonedKernel = *CloneFunction(&Kernel, VMap);
  ClonedKernel.setName(getNoLoopSymbolName(Kernel));

  CallInst *DistributeStaticInit = getFirstKmpcDistributeStaticInitCall(ClonedKernel);
  assert(DistributeStaticInit && "expected the kernel to contain a __kmpc_distribute_static_init_* call");
  Value &TripCount = adjustOuterStaticDistribution(*DistributeStaticInit);

  CallInst *Parallel = getFirstKmpcParallel60Call(ClonedKernel);
  assert(Parallel && "expected the kernel to contain a __kmpc_parallel_60 call");

  IRBuilder<> Builder(Parallel);
  Value *CapturedVarsAddr = Parallel->getArgOperand(7);
  assert(CapturedVarsAddr);
  Builder.CreateStore(Builder.CreateZExt(&TripCount, Builder.getInt64Ty()), CapturedVarsAddr);

  Function *Microtask = getMicrotask(*Parallel);
  assert(Microtask && "expected the __kmpc_parallel_60 to contain a microtask");
  Parallel->setArgOperand(5, &specializeMicrotaskNoLoop(*Microtask));
  
  return ClonedKernel;
}

Function *trySpecializeNoLoop(Function &F) {
  if (!F.hasKernelCallingConv())
    return nullptr;

  const Module &M = *F.getParent();

  const StringRef NoLoopSymbol = getNoLoopSymbolName(F);
  if (M.getNamedValue(NoLoopSymbol)) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped no-loop specialization for kernel `"
                      << F.getName() << "` because `" << NoLoopSymbol << "` already exists\n");
    return nullptr;
  }

  if (!isDistributeParallelFor(F)) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped no-loop specialization for kernel `"
                      << F.getName() << "` because it does not have distribute parallel for structure\n");
    return nullptr;
  }

  return &specializeKernelNoLoop(F);
}

} // namespace


PreservedAnalyses GPUKernelVersioningPass::run(Module &M,
                                               ModuleAnalysisManager &) {
  if (!M.getTargetTriple().isGPU())
    return PreservedAnalyses::all();

  bool Changed = false;
  SmallVector<std::reference_wrapper<Function>, 16> Worklist(M.begin(), M.end());

  for (Function &F : Worklist)
    Changed |= trySpecializeNoLoop(F) != nullptr;

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
