; RUN: opt -S -passes='materialize-kernel-info' < %s | FileCheck %s

; This test checks if kernel info is correctly materialized for different
; kernels.

; It is necessary to specify a GPU target. Otherwise, this pass is a no-op.
target triple = "amdgcn-amd-amdhsa"

; CHECK-DAG: @cuda_kernel_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"i32\00"
define ptx_kernel void @cuda_kernel(i32 %x) {
  ret void
}

; CHECK-DAG: @hip_kernel_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"i64\00"
define amdgpu_kernel void @hip_kernel(i64 %x) {
  ret void
}

; CHECK-DAG: @__omp_offloading_nvptx_l42_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"ptr\00"
define ptx_kernel void @__omp_offloading_nvptx_l42(ptr %dyn) "kernel" {
  ret void
}

; CHECK-DAG: @__omp_offloading_amdgpu_l42_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"ptr\00"
define amdgpu_kernel void @__omp_offloading_amdgpu_l42(ptr %dyn) "kernel" {
  ret void
}

; CHECK-DAG: @spir_kernel_kernel_info = protected unnamed_addr addrspace(1) constant [6 x i8] c"float\00"
define spir_kernel void @spir_kernel(float %x) {
  ret void
}

; CHECK-DAG: @typed_kernel_kernel_info = protected unnamed_addr addrspace(1) constant [25 x i8] c"i32;i64;float;double;ptr\00"
define amdgpu_kernel void @typed_kernel(i32 %i32, i64 %i64, float %f32,
                                        double %f64, ptr %p) {
  ret void
}

; CHECK-DAG: @vector_unknown_kernel_kernel_info = protected unnamed_addr addrspace(1) constant [8 x i8] c"unknown\00"
define amdgpu_kernel void @vector_unknown_kernel(<2 x i32> %v) {
  ret void
}

; CHECK-DAG: @preexisting_kernel_info_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"i32\00"
@preexisting_kernel_info_kernel_info = protected unnamed_addr addrspace(1) constant [4 x i8] c"i32\00"

define amdgpu_kernel void @preexisting_kernel_info(i32 %x) {
  ret void
}

; CHECK-DAG: @zero_initialized_kernel_info = protected unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer
@zero_initialized_kernel_info = protected unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer

define amdgpu_kernel void @zero_initialized() {
  ret void
}
