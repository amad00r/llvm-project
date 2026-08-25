; RUN: split-file %s %t
; RUN: not --crash opt -S -passes='materialize-kernel-info' < %t/not-global.ll 2>&1 | FileCheck %s --check-prefix=NOT-GLOBAL
; RUN: not --crash opt -S -passes='materialize-kernel-info' < %t/no-initializer.ll 2>&1 | FileCheck %s --check-prefix=NO-INIT
; RUN: not --crash opt -S -passes='materialize-kernel-info' < %t/non-data-array.ll 2>&1 | FileCheck %s --check-prefix=NON-DATA-ARRAY
; RUN: not --crash opt -S -passes='materialize-kernel-info' < %t/non-c-string.ll 2>&1 | FileCheck %s --check-prefix=NON-C-STRING
; RUN: not --crash opt -S -passes='materialize-kernel-info' < %t/incompatible.ll 2>&1 | FileCheck %s --check-prefix=INCOMPATIBLE

;--- not-global.ll
target triple = "amdgcn-amd-amdhsa"

define void @not_global_kernel_info() {
  ret void
}

define amdgpu_kernel void @not_global(i32 %x) {
  ret void
}

;--- no-initializer.ll
target triple = "amdgcn-amd-amdhsa"

@no_initializer_kernel_info = external addrspace(1) constant [4 x i8]

define amdgpu_kernel void @no_initializer(i32 %x) {
  ret void
}

;--- non-data-array.ll
target triple = "amdgcn-amd-amdhsa"

@non_data_array_kernel_info = protected addrspace(1) constant { i32 } { i32 0 }

define amdgpu_kernel void @non_data_array(i32 %x) {
  ret void
}

;--- non-c-string.ll
target triple = "amdgcn-amd-amdhsa"

@non_c_string_kernel_info = protected addrspace(1) constant [3 x i8] c"i32"

define amdgpu_kernel void @non_c_string(i32 %x) {
  ret void
}

;--- incompatible.ll
target triple = "amdgcn-amd-amdhsa"

@incompatible_kernel_info = protected addrspace(1) constant [8 x i8] c"i32;i64\00"

define amdgpu_kernel void @incompatible(i32 %x, i32 %y) {
  ret void
}

; NOT-GLOBAL: LLVM ERROR: MaterializeKernelInfo: cannot materialize kernel info because the `not_global_kernel_info` symbol already exists and contains malformed or incompatible information
; NO-INIT: LLVM ERROR: MaterializeKernelInfo: cannot materialize kernel info because the `no_initializer_kernel_info` symbol already exists and contains malformed or incompatible information
; NON-DATA-ARRAY: LLVM ERROR: MaterializeKernelInfo: cannot materialize kernel info because the `non_data_array_kernel_info` symbol already exists and contains malformed or incompatible information
; NON-C-STRING: LLVM ERROR: MaterializeKernelInfo: cannot materialize kernel info because the `non_c_string_kernel_info` symbol already exists and contains malformed or incompatible information
; INCOMPATIBLE: LLVM ERROR: MaterializeKernelInfo: cannot materialize kernel info because the `incompatible_kernel_info` symbol already exists and contains malformed or incompatible information
