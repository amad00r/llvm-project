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
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/Utils/MaterializedKernelInfo.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <variant>

#define DEBUG_TYPE "materialize-kernel-info"

using namespace llvm;

namespace {

std::string getKernelInfoSymbolName(const Function &F) {
  return (F.getName() + "_kernel_info").str();
}

struct LogicalSignatureInference {
  struct ExtendedType {
    std::variant<ExtendedType *, Type *> Variant;

    ExtendedType *unwrap() const {
      if (ExtendedType *const *Inner = std::get_if<ExtendedType *>(&Variant))
        return *Inner;
      return nullptr;
    }
    Type *getLLVMType() const {
      if (Type *const *Ty = std::get_if<Type *>(&this->Variant))
        return *Ty;
      return nullptr;
    }
    bool isTransparentPointerTy() const {
      return std::holds_alternative<ExtendedType *>(Variant);
    }
    static bool isStructurallyEqual(ExtendedType *ET1, ExtendedType *ET2) {
      Type *T1 = ET1->getLLVMType();
      Type *T2 = ET2->getLLVMType();
      if (T1 != T2) return false;
      if (T1) return true;
      assert(ET1->isTransparentPointerTy());
      assert(ET2->isTransparentPointerTy());
      return isStructurallyEqual(ET1->unwrap(), ET2->unwrap());
    }

    std::string dump() const {
      struct Visitor {
        std::string operator()(ExtendedType *Ptr) const {
          return Ptr->dump() + "*";
        }
        std::string operator()(Type *Ty) const {
          std::string S;
          raw_string_ostream RSO(S);
          Ty->print(RSO);
          RSO.flush();
          return S;
        }
      };
      return std::visit(Visitor{}, Variant);
    }
  };

  SpecificBumpPtrAllocator<ExtendedType> ArenaAllocator;
  SmallDenseMap<const Value *, ExtendedType *, 32> ValueTypeMap;

  ExtendedType *getExtType(Type *Ty) {
    if (!Ty) return nullptr;
    return new (ArenaAllocator.Allocate()) ExtendedType{ Ty };
  }
  ExtendedType *getWrapExtType(ExtendedType *Ty) {
    if (!Ty) return nullptr;
    return new (ArenaAllocator.Allocate()) ExtendedType{ Ty };
  }

  ExtendedType *recInstruction(const Use &U) {
    const Value *V = U.get();
    const Instruction *I = cast<Instruction>(U.getUser());
    switch (I->getOpcode()) {
      case Instruction::Load:
        return getWrapExtType(rec(I));
      case Instruction::Store: {
        const auto &SI = *cast<StoreInst>(I);
        if (V != SI.getValueOperand())
          return getWrapExtType(rec(SI.getValueOperand()));
        assert(V == SI.getValueOperand());
        const ExtendedType *ExtTy = rec(SI.getPointerOperand());
        if (!ExtTy)
          return nullptr;
        if (Type *Ty = ExtTy->getLLVMType()) {
          assert(Ty->isPointerTy());
          return getExtType(V->getType());
        }
        if (ExtendedType *Inner = ExtTy->unwrap())
          return Inner;
        llvm_unreachable("Operand 1 must be a pointer");
        return getExtType(V->getType());
      }
      case Instruction::BitCast: case Instruction::AddrSpaceCast:
        return rec(I);
      case Instruction::Call: {
        const Function *F = cast<CallInst>(I)->getCalledFunction();
        if (!F)
          return getExtType(V->getType());
        return rec(F->getArg(U.getOperandNo()));
      }
      default:
        return getExtType(V->getType());
    }
  }

  // ExtendedType *recIntrinsic(const Value *V, const IntrinsicInst *I) {
  //   switch (I->getIntrinsicID()) {
  //     case Intrinsic::dbg_value:
  //       return nullptr;
  //     default:
  //       return getExtType(V->getType());
  //   }
  // }

  ExtendedType *rec(const Value *V) {
    const auto It = ValueTypeMap.find(V);
    if (It != ValueTypeMap.end())
      return It->second;
    ValueTypeMap[V] = nullptr;

    // The logical type of a value is its implementation type by default
    SmallVector<ExtendedType *, 8> ReinterpretSet;

    // Try to refine the logical type based on the uses of the value
    for (const Use &U : V->uses()) {
      ExtendedType *Ty;
      if (isa<Instruction>(U.getUser()))
        Ty = recInstruction(U);
      else
        Ty = getExtType(V->getType());

      if (Ty) {
        ReinterpretSet.emplace_back(Ty);
        LLVM_DEBUG(
          V->printAsOperand(dbgs(), false);
          // TODO: make identifiers be distinguishable interprocedurally.
          dbgs() << " has an estimated type <" << Ty->dump() << "> in instruction { " << *U.getUser() << " }\n"
        );
      }
    }

    if (ReinterpretSet.empty()) {
      ExtendedType *Ty = getExtType(V->getType());
      ValueTypeMap[V] = Ty;
      LLVM_DEBUG(
        V->printAsOperand(dbgs(), false);
        dbgs() << " falls back to implementation type <" << Ty->dump() << "> as it does not have available uses\n"
      );
      return Ty;
    }

    for (size_t i = 1; i < ReinterpretSet.size(); ++i)
      if (!ExtendedType::isStructurallyEqual(ReinterpretSet[0], ReinterpretSet[i])) {
        ExtendedType *Ty = getExtType(V->getType());
        ValueTypeMap[V] = Ty;
        LLVM_DEBUG(
          V->printAsOperand(dbgs(), false);
          dbgs() << " falls back to implementation type <" << Ty->dump() << "> to be conservative\n"
        );
        return Ty;
      }

    ExtendedType *Ty = ReinterpretSet[0];
    LLVM_DEBUG(
      V->printAsOperand(dbgs(), false);
      dbgs() << " has an estimated logical type <" << Ty->dump() << ">\n"
    );
    ValueTypeMap[V] = Ty;
    return Ty;
  }

  SmallVector<ExtendedType *, 16> operator()(const Function &F) {
    LLVM_DEBUG(dbgs() << F << "\n");
    SmallVector<ExtendedType *, 16> LogicalArgs;
    for (const Argument &Arg : F.args())
      LogicalArgs.emplace_back(rec(&Arg));
    return LogicalArgs;
  }
};

KernelArgTypeInfo
getKernelArgTypeInfo(Type *Ty, int Indirection = 0) {
  assert(Ty);
  if (Ty->isIntegerTy())
    return KernelArgTypeInfo::getIntegerTy(
        cast<IntegerType>(Ty)->getBitWidth(), Indirection);
  if (Ty->isFloatTy())
    return KernelArgTypeInfo::getFloatTy(Indirection);
  if (Ty->isDoubleTy())
    return KernelArgTypeInfo::getDoubleTy(Indirection);
  if (Ty->isPointerTy())
    return KernelArgTypeInfo::getOpaquePointerTy(Indirection);
  return KernelArgTypeInfo::getUnknownTy();
}

KernelArgTypeInfo
getKernelArgTypeInfo(LogicalSignatureInference::ExtendedType *ExtTy) {
  uint32_t Indirection = 0;
  while (ExtTy->isTransparentPointerTy()) {
    ++Indirection;
    ExtTy = ExtTy->unwrap();
  }
  return getKernelArgTypeInfo(ExtTy->getLLVMType(), Indirection);
}

template <typename T>
void appendTypeInfoOf(T *Ty, SmallVector<uint8_t, sizeof(KernelArgTypeInfo) * 16> &EncodedKernelInfo) {
  const KernelArgTypeInfo TypeInfo = getKernelArgTypeInfo(Ty);
  const uint8_t *Begin = reinterpret_cast<const uint8_t *>(&TypeInfo);
  EncodedKernelInfo.append(Begin, Begin + sizeof(TypeInfo));
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
  if (omp::isOpenMPKernel(F)) {
    LogicalSignatureInference LSI;
    for (LogicalSignatureInference::ExtendedType *ArgType : LSI(F))
      appendTypeInfoOf(ArgType, EncodedKernelInfo);
  } else {
    for (const Argument &Arg : F.args())
      appendTypeInfoOf(Arg.getType(), EncodedKernelInfo);
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
