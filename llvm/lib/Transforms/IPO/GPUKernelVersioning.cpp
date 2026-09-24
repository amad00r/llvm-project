//===- GPUKernelVersioning.cpp - GPU kernel versioning pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/GPUKernelVersioning.h"
#include "llvm/Transforms/Utils/MaterializedKernelInfo.h"
#include "llvm/Transforms/Utils/MaterializeKernelInfo.h"
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

cl::opt<bool> DisableNoLoopVersioning(
    "gpu-kernel-versioning-disable-noloop",
    cl::desc("Disable the generation of .noloop versions."),
    cl::Hidden,
    cl::init(false)
);
cl::opt<bool> DisableNoAliasVersioning(
    "gpu-kernel-versioning-disable-noalias",
    cl::desc("Disable the generation of .noalias versions."),
    cl::Hidden,
    cl::init(false)
);
cl::opt<bool> DisableAlign16Versioning(
    "gpu-kernel-versioning-disable-align16",
    cl::desc("Disable the generation of .align16 versions."),
    cl::Hidden,
    cl::init(false)
);
cl::opt<bool> DisableUnroll2Versioning(
    "gpu-kernel-versioning-disable-unroll2",
    cl::desc("Disable the generation of .unroll2 versions."),
    cl::Hidden,
    cl::init(false)
);

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

CallInst *getFirstCallThatEndsWith(Function &F, StringRef Suffix) {
  return getFirstCallThatSatisfies(F, [Suffix](const CallInst &CI) {
    if (const Function *Callee = CI.getCalledFunction())
      return Callee->getName().ends_with(Suffix);
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
CallInst *getFirstOutlineCall(Function &F) {
  return getFirstCallThatEndsWith(F, "_omp_outlined");
}

Function *getMicrotask(const CallInst &KmpcParallel60Call) {
  return cast<Function>(KmpcParallel60Call.getArgOperand(5)->stripPointerCasts());
}

bool isDistributeParallelFor(Function &F) {
  const CallInst *Outline = getFirstOutlineCall(F);
  if (!Outline) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: kernel `" << F.getName() << "` does not call `" << F.getName() << "_omp_outlined`\n");
    return false;
  }
  Function *OutlineFn = Outline->getCalledFunction();
  assert(OutlineFn);
  const CallInst *Outer = getFirstKmpcDistributeStaticInitCall(*OutlineFn);
  if (!Outer) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: helper `" << OutlineFn->getName() << "` does not call `__kmpc_distribute_static_init_*`\n");
    return false;
  }
  const CallInst *Parallel = getFirstKmpcParallel60Call(*OutlineFn);
  if (!Parallel) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: kernel `" << OutlineFn->getName() << "` does not call `__kmpc_parallel_60`\n");
    return false;
  }
  Function *Microtask = getMicrotask(*Parallel);
  if (!Microtask) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: kernel `" << OutlineFn->getName() << "` calls `__kmpc_parallel_60` but has malformed microtask\n");
    return false;
  }
  const CallInst *Inner = getFirstKmpcForStaticInitCall(*Microtask);
  if (!Inner) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: microtask `" << Microtask->getName() << "` does not call `__kmpc_for_static_init_*`\n");
    return false;
  }
  return true;
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
  Function &SpecializedMicrotask = *CloneFunction(&Microtask, VMap);

  CallInst *ForStaticInit = getFirstKmpcForStaticInitCall(SpecializedMicrotask);
  assert(ForStaticInit && "expected the microtask to contain a __kmpc_for_static_init_* call");

  adjustInnerStaticDistribution(*ForStaticInit);
  return SpecializedMicrotask;
}

void specializeOutlineNoLoop(Function &Outline) {
  CallInst *DistributeStaticInit = getFirstKmpcDistributeStaticInitCall(Outline);
  assert(DistributeStaticInit && "expected the kernel to contain a __kmpc_distribute_static_init_* call");
  Value &TripCount = adjustOuterStaticDistribution(*DistributeStaticInit);

  CallInst *Parallel = getFirstKmpcParallel60Call(Outline);
  assert(Parallel && "expected the kernel to contain a __kmpc_parallel_60 call");

  IRBuilder<> Builder(Parallel);
  Value *CapturedVarsAddr = Parallel->getArgOperand(7);
  assert(CapturedVarsAddr);
  Builder.CreateStore(Builder.CreateZExt(&TripCount, Builder.getInt64Ty()), CapturedVarsAddr);

  Function *Microtask = getMicrotask(*Parallel);
  assert(Microtask && "expected the __kmpc_parallel_60 to contain a microtask");
  Parallel->setArgOperand(5, &specializeMicrotaskNoLoop(*Microtask));
}

SmallString<64> getNoLoopSymbolName(const Function &F) {
  // TODO: get the .noloop from MaterializedKernelInfo
  return { F.getName(), ".noloop" };
}
SmallString<64> getNoAliasSymbolName(const Function &F) {
  // TODO: get the .noalias from MaterializedKernelInfo
  return { F.getName(), ".noalias" };
}
SmallString<64> getAlign16SymbolName(const Function &F) {
  // TODO: get the .align16 from MaterializedKernelInfo
  return { F.getName(), ".align16" };
}
SmallString<64> getUnroll2SymbolName(const Function &F) {
  // TODO: get the .align16 from MaterializedKernelInfo
  return { F.getName(), ".unroll2" };
}

void cloneKernelEnvironmentForVersion(Function &Original, Function &Version) {
  Module &M = *Original.getParent();
  assert(&M == Version.getParent());

  const SmallString<128> OriginalKernelEnvSymbol{ Original.getName(), "_kernel_environment" };
  GlobalVariable *OriginalKernelEnvGV = M.getNamedGlobal(OriginalKernelEnvSymbol);
  assert(OriginalKernelEnvGV);

  const SmallString<128> VersionKernelEnvSymbol{ Version.getName(), "_kernel_environment" };
  assert(!M.getNamedGlobal(VersionKernelEnvSymbol));
  GlobalVariable *VersionKernelEnvGV = new GlobalVariable(
    M,
    OriginalKernelEnvGV->getValueType(),
    OriginalKernelEnvGV->isConstant(),
    OriginalKernelEnvGV->getLinkage(),
    OriginalKernelEnvGV->hasInitializer() ? OriginalKernelEnvGV->getInitializer() : nullptr,
    VersionKernelEnvSymbol,
    nullptr,
    OriginalKernelEnvGV->getThreadLocalMode(),
    OriginalKernelEnvGV->getType()->getAddressSpace(),
    OriginalKernelEnvGV->isExternallyInitialized()
  );
  assert(VersionKernelEnvGV);
  VersionKernelEnvGV->copyAttributesFrom(OriginalKernelEnvGV);
}

Function *prepareFirstOutlineForSpecialization(Function &F) {
  const Twine SpecializedOutlineName{ F.getName(), "_omp_outlined" };
  assert(!F.getParent()->getNamedValue(SpecializedOutlineName.str()));
  CallInst *Outline = getFirstOutlineCall(F);
  if (!Outline)
    return nullptr;
  Function *OutlineFn = Outline->getCalledFunction();
  assert(OutlineFn);
  ValueToValueMapTy VMap;
  Function &ClonedOutlineFn = *CloneFunction(OutlineFn, VMap);
  ClonedOutlineFn.setName(SpecializedOutlineName);
  Outline->setCalledFunction(&ClonedOutlineFn);
  return &ClonedOutlineFn;
}

Function &specializeNoLoop(Function &Kernel) {
  ValueToValueMapTy VMap;
  Function &SpecializedKernel = *CloneFunction(&Kernel, VMap);
  SpecializedKernel.setName(getNoLoopSymbolName(Kernel));

  Function *SpecializedOutline = prepareFirstOutlineForSpecialization(SpecializedKernel);
  assert(SpecializedOutline);
  specializeOutlineNoLoop(*SpecializedOutline);

  return SpecializedKernel;
}

Function *trySpecializeNoLoop(Function &F) {
  assert(F.hasKernelCallingConv());

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

  return &specializeNoLoop(F);
}

void addNoAliasToPtrArgs(Function &F) {
  for (auto &Arg : F.args())
    if (Arg.getType()->isPointerTy())
        Arg.addAttr(Attribute::NoAlias);
}

Function &cloneAndAddNoAliasToPtrArgs(Function &F) {
  ValueToValueMapTy VMap;
  Function &Clone = *CloneFunction(&F, VMap);
  addNoAliasToPtrArgs(Clone);
  return Clone;
}

Function &specializeNoAlias(Function &Kernel) {
  Function &SpecializedKernel = cloneAndAddNoAliasToPtrArgs(Kernel);
  SpecializedKernel.setName(getNoAliasSymbolName(Kernel));

  Function *SpecializedOutline = prepareFirstOutlineForSpecialization(SpecializedKernel);
  if (!SpecializedOutline)
    return SpecializedKernel;
  addNoAliasToPtrArgs(*SpecializedOutline);
  
  if (CallInst *Parallel = getFirstKmpcParallel60Call(*SpecializedOutline)) {
    Function *Microtask = getMicrotask(*Parallel);
    assert(Microtask && "expected the __kmpc_parallel_60 to contain a microtask");
    Function &SpecializedMicrotask = cloneAndAddNoAliasToPtrArgs(*Microtask);
    // const Twine SpecializedMicrotaskName{ SpecializedOutline.getName(), "_omp_outlined" };
    // assert(!SpecializedMicrotask.getParent()->getNamedValue(SpecializedMicrotaskName));
    // SpecializedMicrotask.setName(SpecializedMicrotaskName);
    Parallel->setArgOperand(5, &SpecializedMicrotask);
  }

  return SpecializedKernel;
}

Function *trySpecializeNoAlias(Function &F) {
  assert(F.hasKernelCallingConv());

  const Module &M = *F.getParent();

  const StringRef NoAliasSymbol = getNoAliasSymbolName(F);
  if (M.getNamedValue(NoAliasSymbol)) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped noalias specialization for kernel `"
                      << F.getName() << "` because `" << NoAliasSymbol << "` already exists\n");
    return nullptr;
  }

  if (!std::any_of(F.args().begin(), F.args().end(), [](const Argument &Arg) { return Arg.getType()->isPointerTy() && !Arg.hasNoAliasAttr(); })) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped noalias specialization for kernel `"
                      << F.getName() << "` there are no ptr args, or all of them are noalias already\n");
    return nullptr;
  }

  return &specializeNoAlias(F);
}

KernelInfo createKernelInfoWith0Versions() {
  KernelInfo KI;
  KI.Versions.emplace();
  return KI;
}

// TODO: assuming last argument is dyn is dangerous. For previous versions it was different
void addAlign16ToPtrArgsExceptDyn(Function &F) {
  for (size_t I = 0; I < F.arg_size() - 1; ++I) {
    auto &Arg = *F.getArg(I);
    if (Arg.getType()->isPointerTy())
        Arg.addAttr(Attribute::getWithAlignment(Arg.getContext(), Align(16)));
  }
}

Function &cloneAndAddAlign16ToPtrArgsExceptDyn(Function &F) {
  ValueToValueMapTy VMap;
  Function &Clone = *CloneFunction(&F, VMap);
  addAlign16ToPtrArgsExceptDyn(Clone);
  return Clone;
}

// TODO: assuming first 2 are the global tid and bound tid is fragile
void addAlign16ToPtrArgsExceptFirst2(Function &F) {
  for (size_t I = 2; I < F.arg_size(); ++I) {
    auto &Arg = *F.getArg(I);
    if (Arg.getType()->isPointerTy())
        Arg.addAttr(Attribute::getWithAlignment(Arg.getContext(), Align(16)));
  }
}

Function &cloneAndAddAlign16ToPtrArgsExceptFirst2(Function &F) {
  ValueToValueMapTy VMap;
  Function &Clone = *CloneFunction(&F, VMap);
  addAlign16ToPtrArgsExceptFirst2(Clone);
  return Clone;
}

Function &specializeAlign16(Function &Kernel) {
  Function &SpecializedKernel = cloneAndAddAlign16ToPtrArgsExceptDyn(Kernel);
  SpecializedKernel.setName(getAlign16SymbolName(Kernel));

  Function *SpecializedOutline = prepareFirstOutlineForSpecialization(SpecializedKernel);
  if (!SpecializedOutline)
    return SpecializedKernel;
  // TODO: I assume the first outline has same signature as the kernel, but anyways it is not required to add the attr here i think
  // addAlign16ToPtrArgsExceptDyn(*SpecializedOutline);

  if (CallInst *Parallel = getFirstKmpcParallel60Call(*SpecializedOutline)) {
    Function *Microtask = getMicrotask(*Parallel);
    assert(Microtask && "expected the __kmpc_parallel_60 to contain a microtask");
    Function &SpecializedMicrotask = cloneAndAddAlign16ToPtrArgsExceptFirst2(*Microtask);
    Parallel->setArgOperand(5, &SpecializedMicrotask);
  }

  return SpecializedKernel;
}

Function *trySpecializeAlign16(Function &F) {
  assert(F.hasKernelCallingConv());

  const Module &M = *F.getParent();

  const StringRef Align16Symbol = getAlign16SymbolName(F);
  if (M.getNamedValue(Align16Symbol)) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped align16 specialization for kernel `"
                      << F.getName() << "` because `" << Align16Symbol << "` already exists\n");
    return nullptr;
  }

  // TODO: codegen does not add alignment attributes to kernel args no?

  return &specializeAlign16(F);
}

Function &specializeUnroll2(Function &Kernel) {
  // TODO: implement. this is a noop right now, the kernel is cloned but not transformed
  ValueToValueMapTy VMap;
  Function &SpecializedKernel = *CloneFunction(&Kernel, VMap);
  SpecializedKernel.setName(getUnroll2SymbolName(F));
  return SpecializedKernel;
}

Function *trySpecializeUnroll2(Function &F) {
  assert(F.hasKernelCallingConv());

  const Module &M = *F.getParent();

  const StringRef Unroll2Symbol = getUnroll2SymbolName(F);
  if (M.getNamedValue(Unroll2Symbol)) {
    LLVM_DEBUG(dbgs() << "GPUKernelVersioning: skipped unroll2 specialization for kernel `"
                      << F.getName() << "` because `" << Unroll2Symbol << "` already exists\n");
    return nullptr;
  }

  return &specializeUnroll2(F);
}

} // namespace

PreservedAnalyses GPUKernelVersioningPass::run(Module &M,
                                               ModuleAnalysisManager &) {
  if (!M.getTargetTriple().isGPU())
    return PreservedAnalyses::all();

  SmallVector<std::pair<KernelInfo, std::reference_wrapper<Function>>, 16> Worklist;
  for (Function &F : M) {
    if (!F.hasKernelCallingConv())
      continue;
    if (auto KI = kernel_info_utils::parseGlobal(M, KernelInfo::getGlobalNameFor(F.getName()))) {
      if (KI->Versions)
        continue;
      auto &Version = Worklist.emplace_back(*KI, F);
      Version.first.Versions.emplace();
    } else {
      Worklist.emplace_back(createKernelInfoWith0Versions(), F);
    }
  }

  if (!DisableAlign16Versioning)
    for (size_t Size = Worklist.size(), I = 0; I < Size; ++I) {
      auto &[KI, Kernel] = Worklist[I];
      if (Function *Version = trySpecializeAlign16(Kernel)) {
        cloneKernelEnvironmentForVersion(Kernel, *Version);
        KI.Versions->push_back({ { KernelInfo::Version::Specialization::Align16{} } });
        Worklist.emplace_back(createKernelInfoWith0Versions(), *Version);
      }
    }

  if (!DisableNoAliasVersioning)
    for (size_t Size = Worklist.size(), I = 0; I < Size; ++I) {
      auto &[KI, Kernel] = Worklist[I];
      if (Function *Version = trySpecializeNoAlias(Kernel)) {
        cloneKernelEnvironmentForVersion(Kernel, *Version);
        KI.Versions->push_back({ { KernelInfo::Version::Specialization::NoAlias{} } });
        Worklist.emplace_back(createKernelInfoWith0Versions(), *Version);
      }
    }

  if (!DisableUnroll2Versioning)
    for (size_t Size = Worklist.size(), I = 0; I < Size; ++I) {
      auto &[KI, Kernel] = Worklist[I];
      if (Function *Version = trySpecializeUnroll2(Kernel)) {
        cloneKernelEnvironmentForVersion(Kernel, *Version);
        KI.Versions->push_back({ { KernelInfo::Version::Specialization::Unroll2{} } });
        Worklist.emplace_back(createKernelInfoWith0Versions(), *Version);
      }
    }

  if (!DisableNoLoopVersioning)
    for (size_t Size = Worklist.size(), I = 0; I < Size; ++I) {
      auto &[KI, Kernel] = Worklist[I];
      if (Function *Version = trySpecializeNoLoop(Kernel)) {
        cloneKernelEnvironmentForVersion(Kernel, *Version);
        KI.Versions->push_back({ { KernelInfo::Version::Specialization::NoLoop{} } });
        Worklist.emplace_back(createKernelInfoWith0Versions(), *Version);
      }
    }

  for (const auto &[KI, Version] : Worklist)
    kernel_info_utils::createOrReplaceGlobal(M, KernelInfo::getGlobalNameFor(Version.get().getName()), KI);

  // TODO: maybe we can relax it
  return PreservedAnalyses::none();
}
