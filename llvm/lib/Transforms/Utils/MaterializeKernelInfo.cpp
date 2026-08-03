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

std::optional<KernelInfo> parseGlobalFor(Function &Kernel) {
  assert(Kernel.hasKernelCallingConv());

  GlobalValue *GV = Kernel.getParent()->getNamedValue(KernelInfo::getGlobalNameFor(Kernel.getName()));
  if (!GV)
    return {};

  const auto &GVar = *cast<GlobalVariable>(GV);
  const Constant *Init = GVar.getInitializer();
  assert(Init);

  if (Init->isNullValue())
    return KernelInfo{};

  return { cast<ConstantDataArray>(Init)->getAsCString() };
}

namespace {

void createGlobalFor(Module &M, StringRef KISymbol, GlobalVariable::LinkageTypes Linkage, GlobalVariable::VisibilityTypes Visibility, const KernelInfo &KI) {
  assert(!M.getNamedValue(KISymbol));

  IRBuilder<> Builder(M.getContext());
  GlobalVariable &KIGVar = *Builder.CreateGlobalString(KI.str(), KISymbol, M.getDataLayout().getDefaultGlobalsAddressSpace(), &M, true);
  KIGVar.setLinkage(Linkage);
  KIGVar.setVisibility(Visibility);
}

} // namespace

void createGlobalFor(Function &Kernel, const KernelInfo &KI) {
  assert(Kernel.hasKernelCallingConv());
  createGlobalFor(*Kernel.getParent(), KernelInfo::getGlobalNameFor(Kernel.getName()), Kernel.getLinkage(), Kernel.getVisibility(), KI);
}

void createOrReplaceGlobalFor(Function &Kernel, const KernelInfo &KI) {
  assert(Kernel.hasKernelCallingConv());

  Module &M = *Kernel.getParent();

  const auto KISymbol = KernelInfo::getGlobalNameFor(Kernel.getName());
  if (GlobalValue *GV = M.getNamedValue(KISymbol)) {
    auto &GVar = *cast<GlobalVariable>(GV);
    assert(GVar.hasInitializer());
    assert(GVar.getInitializer());
    assert(GVar.getInitializer()->isNullValue() || cast<ConstantDataArray>(GVar.getInitializer())->isCString());
    GVar.replaceInitializer(ConstantDataArray::getString(M.getContext(), KI.str(), true));

    SmallVector<std::reference_wrapper<GlobalAlias>, 4> Aliases;
    for (User *Usr : GVar.users())
      if (GlobalAlias *GA = dyn_cast<GlobalAlias>(Usr))
        Aliases.emplace_back(*GA);

    for (GlobalAlias &Alias : Aliases) {
      GlobalAlias *NewAlias = GlobalAlias::create(
        GVar.getValueType(),
        Alias.getAddressSpace(),
        Alias.getLinkage(),
        "",
        Alias.getAliasee(),
        &M
      );
      NewAlias->setVisibility(Alias.getVisibility());
      NewAlias->takeName(&Alias);
      // Alias.replaceAllUsesWith(&NewAlias);
      Alias.eraseFromParent();
    }
  } else {
    createGlobalFor(*Kernel.getParent(), KISymbol, Kernel.getLinkage(), Kernel.getVisibility(), KI);
  }
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

  auto KI = kernel_info_utils::parseGlobalFor(F).value_or({});
  assert(!KI.Args || *KI.Args == getKernelInfoArgs(F));
  KI.Args.emplace(getKernelInfoArgs(F));

  kernel_info_utils::createOrReplaceGlobalFor(F, KI);

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

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
