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

namespace {

std::string getKernelInfoSymbolName(const Function &F) {
  return (F.getName() + "_kernel_info").str();
}

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

KernelInfo getKernelInfo(const Function &F) {
  KernelInfo KI;
  for (const Argument &Arg : F.args())
    KI.Args.emplace_back(getKernelInfoArgType(*Arg.getType()));
  return KI;
}

std::optional<KernelInfo> getKernelInfo(const GlobalValue &GV) {
  const auto *GVar = dyn_cast<GlobalVariable>(&GV);
  if (!GVar || !GVar->hasInitializer())
    return {};

  if (GVar->getInitializer()->isNullValue()) {
    const auto *ArrayTy = dyn_cast<ArrayType>(GVar->getValueType());
    if (ArrayTy && ArrayTy->getElementType()->isIntegerTy(8) && ArrayTy->getNumElements() == 1)
      return KernelInfo("");
  }

  const auto *Init = dyn_cast<ConstantDataArray>(GVar->getInitializer());
  if (!Init || !Init->isCString())
    return {};

  return KernelInfo(Init->getAsCString());
}

bool materializeKernelInfo(Function &F, IRBuilder<> &Builder) {
  if (!F.hasKernelCallingConv())
    return false;

  Module &M = *F.getParent();

  const std::string KernelInfoSymbol = getKernelInfoSymbolName(F);
  const KernelInfo KI = getKernelInfo(F);
  if (M.getNamedValue(KernelInfoSymbol)) {
    const std::optional<KernelInfo> ExistingKI = getKernelInfo(*M.getNamedValue(KernelInfoSymbol));
    if (!ExistingKI || KI != *ExistingKI)
      report_fatal_error(
        Twine("MaterializeKernelInfo: cannot materialize kernel info because the `") +
        KernelInfoSymbol +
        "` symbol already exists and contains malformed or incompatible information",
        /*gen_crash_diag=*/true);
    return false;
  }

  GlobalVariable &KernelInfoGV = *Builder.CreateGlobalString(KI.str(), KernelInfoSymbol, M.getDataLayout().getDefaultGlobalsAddressSpace(), &M, true);
  KernelInfoGV.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelInfoGV.setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  appendToCompilerUsed(M, { &KernelInfoGV });

  return true;
}

} // namespace

PreservedAnalyses MaterializeKernelInfoPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
  if (!M.getTargetTriple().isGPU())
    return PreservedAnalyses::all();

  IRBuilder<> Builder(M.getContext());

  bool Changed = false;
  for (Function &F : M)
    Changed |= materializeKernelInfo(F, Builder);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
