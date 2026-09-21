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
#include "llvm/Transforms/Utils/MaterializedKernelInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/IR/IRBuilder.h"
#include <cassert>
#include <optional>
#include <string>

using namespace llvm;

namespace llvm::kernel_info_utils {

// TODO: we want to change this. instead of removing first, remove later if what we want to create has changed

std::optional<KernelInfo> parseAndEraseGlobal(Module &M, StringRef KISymbol) {
  GlobalValue *GV = M.getNamedValue(KISymbol);
  if (!GV)
    return {};

  const auto *GVar = dyn_cast<GlobalVariable>(GV);
  assert(GVar);
  assert(GVar->hasInitializer());

  const Constant *Init = GVar->getInitializer();
  assert(Init);

  if (Init->isNullValue())
    return KernelInfo{};

  const auto *InitArray = dyn_cast<ConstantDataArray>(Init);
  assert(InitArray);

  assert(InitArray->isCString());
  KernelInfo KI(InitArray->getAsCString());
  removeFromUsedLists(M, [&](Constant *C) { return C->hasName() && C->getName() == KISymbol; });
  GV->eraseFromParent();
  return KI;
}

void createGlobal(Module &M, StringRef KISymbol, const KernelInfo &KI) {
  IRBuilder<> Builder(M.getContext());
  GlobalVariable &KernelInfoGV = *Builder.CreateGlobalString(KI.str(), KISymbol, M.getDataLayout().getDefaultGlobalsAddressSpace(), &M, true);
  KernelInfoGV.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelInfoGV.setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  appendToCompilerUsed(M, { &KernelInfoGV });
}

} // end namespace llvm::kernel_info_utils

namespace {

KernelInfo::Argument::Type getKernelInfoArgType(const Type &Ty) {
  switch (Ty.getTypeID()) {
    case Type::IntegerTyID:
      return { KernelInfo::Argument::Type::Int{ cast<IntegerType>(Ty).getBitWidth() } };
    case Type::FloatTyID:
      return { KernelInfo::Argument::Type::Float{} };
    case Type::DoubleTyID:
      return { KernelInfo::Argument::Type::Double{} };
    case Type::PointerTyID:
      return { KernelInfo::Argument::Type::Ptr{} };
    default:
      return { KernelInfo::Argument::Type::Unknown{} };
  }
}

SmallVector<KernelInfo::Argument, 8> getKernelInfoArgs(const Function &F) {
  SmallVector<KernelInfo::Argument, 8> Args;
  for (const Argument &Arg : F.args())
    Args.emplace_back(getKernelInfoArgType(*Arg.getType()));
  return Args;
}

bool materializeKernelInfo(Function &F) {
  if (!F.hasKernelCallingConv())
    return false;

  Module &M = *F.getParent();

  const std::string KISymbol = KernelInfo::getGlobalNameFor(F.getName());
  if (auto MaybeKI = kernel_info_utils::parseAndEraseGlobal(M, KISymbol)) {
    auto &KI = *MaybeKI;
    assert(!KI.Args || *KI.Args == getKernelInfoArgs(F));
    KI.Args.emplace(getKernelInfoArgs(F));
    kernel_info_utils::createGlobal(M, KISymbol, KI);
  } else {
    KernelInfo KI{};
    KI.Args.emplace(getKernelInfoArgs(F));
    kernel_info_utils::createGlobal(M, KISymbol, KI);
  }

  return true;
}

} // namespace

PreservedAnalyses MaterializeKernelInfoPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
  if (!M.getTargetTriple().isGPU())
    return PreservedAnalyses::all();

  bool Changed = false;
  for (Function &F : M)
    Changed |= materializeKernelInfo(F);

  // TODO: be careful, we are erasing the globals unconditionally
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
