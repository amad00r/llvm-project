//===- MaterializeKernelInfo.cpp - Materialize kernel info ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass materializes compile-time kernel information as IR globals so it is
// available at runtime, for example to support profiling.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MaterializeKernelInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/MaterializedKernelInfo.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>
#include <string>

using namespace llvm;

namespace {

std::string getKernelInfoSymbolName(const Function &F) {
  return (F.getName() + "_kernel_info").str();
}

KernelArgTypeInfo getKernelArgTypeInfo(Type *ArgTy) {
  if (ArgTy->isIntegerTy())
    return KernelArgTypeInfo::getIntegerTy(
        cast<IntegerType>(ArgTy)->getBitWidth());
  if (ArgTy->isFloatTy())
    return KernelArgTypeInfo::getFloatTy();
  if (ArgTy->isDoubleTy())
    return KernelArgTypeInfo::getDoubleTy();
  if (ArgTy->isPointerTy())
    return KernelArgTypeInfo::getPointerTy();
  return KernelArgTypeInfo::getUnknownTy();
}

GlobalVariable *createKernelInfoGlobal(Module &M, StringRef Name,
                                       ArrayRef<uint8_t> EncodedKernelInfo) {
  Constant *Init = ConstantDataArray::get(M.getContext(), EncodedKernelInfo);
  auto *Ty =
      ArrayType::get(Type::getInt8Ty(M.getContext()), EncodedKernelInfo.size());
  return new GlobalVariable(M, Ty, /*isConstant=*/true,
                            GlobalValue::ExternalLinkage, Init, Name);
}

bool materializeKernelInfo(Function &F) {
  // FIXME: does not work when target is CPU. we could add a flag to the
  // module to know if it is going through the offloading driver.
  if (!F.hasKernelCallingConv())
    return false;

  const std::string KernelInfoSymbol = getKernelInfoSymbolName(F);
  if (F.getParent()->getNamedValue(KernelInfoSymbol))
    return false;

  SmallVector<uint8_t, sizeof(KernelArgTypeInfo) * 16> EncodedKernelInfo;
  for (const Argument &Arg : F.args()) {
    const KernelArgTypeInfo TypeInfo = getKernelArgTypeInfo(Arg.getType());
    const uint8_t *Begin = reinterpret_cast<const uint8_t *>(&TypeInfo);
    EncodedKernelInfo.append(Begin, Begin + sizeof(TypeInfo));
  }

  GlobalVariable *GV = createKernelInfoGlobal(*F.getParent(), KernelInfoSymbol,
                                              EncodedKernelInfo);
  appendToCompilerUsed(*F.getParent(), {GV});
  return true;
}

} // namespace

PreservedAnalyses MaterializeKernelInfoPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
  bool Changed = false;
  for (Function &F : M)
    Changed |= materializeKernelInfo(F);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
