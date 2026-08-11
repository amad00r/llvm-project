; RUN: opt -S -passes='materialize-kernel-info' < %s | FileCheck %s

; This test checks if kernel info is correctly materialized for `ptx_kernel`,
; `amdgpu_kernel` and `spir_kernel` kernels.

; CHECK-DAG: @cuda_kernel_kernel_info = constant [4 x i8] c"\00 \00\00"
define ptx_kernel void @cuda_kernel(i32 %x) {
  ret void
}

; CHECK-DAG: @hip_kernel_kernel_info = constant [4 x i8] c"\00@\00\00"
define amdgpu_kernel void @hip_kernel(i64 %x) {
  ret void
}

; CHECK-DAG: @__omp_offloading_nvptx_l42_kernel_info = constant [8 x i8] c"\03\00\00\01\00@\00\00"
define weak ptx_kernel void @__omp_offloading_nvptx_l42(ptr %dyn, i64 %x) "kernel" {
  ret void
}

; CHECK-DAG: @__omp_offloading_amdgpu_l42_kernel_info = constant [8 x i8] c"\03\00\00\01\00@\00\00"
define weak amdgpu_kernel void @__omp_offloading_amdgpu_l42(ptr %dyn, i64 %x) "kernel" {
  ret void
}

; CHECK-DAG: @__omp_offloading_amdgpu_l43_kernel_info = constant [4 x i8] c"\00 \00\00"
define weak amdgpu_kernel void @__omp_offloading_amdgpu_l43(i64 %x) "kernel" {
  %x.addr = alloca i64, align 8
  store i64 %x, ptr %x.addr, align 8
  %logical = load i32, ptr %x.addr, align 4
  ret void
}

; CHECK-DAG: @__omp_offloading_nvptx_l44_kernel_info = constant [16 x i8] c"\02\00\00\01\02\00\00\00\02\00\00\00\03\00\00\01"
define weak ptx_kernel void @__omp_offloading_nvptx_l44(ptr noundef nonnull align 8 dereferenceable(8) %0, i64 noundef %1, i64 noundef %2, ptr noalias noundef %3) "kernel" {
  %5 = alloca ptr, align 8, addrspace(5)
  %6 = alloca i64, align 8, addrspace(5)
  %7 = alloca i64, align 8, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = addrspacecast ptr addrspace(5) %5 to ptr
  %10 = addrspacecast ptr addrspace(5) %6 to ptr
  %11 = addrspacecast ptr addrspace(5) %7 to ptr
  %12 = addrspacecast ptr addrspace(5) %8 to ptr
  store ptr %0, ptr %9, align 8
  store i64 %1, ptr %10, align 8
  store i64 %2, ptr %11, align 8
  store ptr %3, ptr %12, align 8
  %13 = load ptr, ptr %9, align 8
  %17 = load double, ptr %10, align 8
  %18 = load double, ptr %11, align 8
  %19 = fadd double %17, %18
  store double %19, ptr %13, align 8
  ret void
}

; CHECK-DAG: @__omp_offloading_amdgpu_l45_kernel_info = constant [16 x i8] c"\02\00\00\01\02\00\00\01\02\00\00\01\03\00\00\01"
define weak_odr protected amdgpu_kernel void @__omp_offloading_amdgpu_l45(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noalias noundef %3) "kernel" {
  %5 = alloca ptr, align 8, addrspace(5)
  %6 = alloca ptr, align 8, addrspace(5)
  %7 = alloca ptr, align 8, addrspace(5)
  %8 = alloca ptr, align 8, addrspace(5)
  %9 = addrspacecast ptr addrspace(5) %5 to ptr
  %10 = addrspacecast ptr addrspace(5) %6 to ptr
  %11 = addrspacecast ptr addrspace(5) %7 to ptr
  %12 = addrspacecast ptr addrspace(5) %8 to ptr
  store ptr %0, ptr %9, align 8
  store ptr %1, ptr %10, align 8
  store ptr %2, ptr %11, align 8
  store ptr %3, ptr %12, align 8
  %13 = load ptr, ptr %9, align 8
  %14 = load ptr, ptr %10, align 8
  %15 = load ptr, ptr %11, align 8
  %19 = load double, ptr %14, align 8
  %20 = load double, ptr %15, align 8
  %21 = fadd double %19, %20
  store double %21, ptr %13, align 8
  ret void
}

; CHECK-DAG: @spir_kernel_kernel_info = constant [4 x i8] c"\01\00\00\00"
define spir_kernel void @spir_kernel(float %x) {
  ret void
}

; CHECK-DAG: @typed_kernel_kernel_info = constant [20 x i8] c"\00 \00\00\00@\00\00\01\00\00\00\02\00\00\00\03\00\00\01"
define amdgpu_kernel void @typed_kernel(i32 %i32, i64 %i64, float %f32,
                                        double %f64, ptr %p) {
  ret void
}

; CHECK-DAG: @vector_unknown_kernel_kernel_info = constant [4 x i8] c"\FF\00\00\00"
define amdgpu_kernel void @vector_unknown_kernel(<2 x i32> %v) {
  ret void
}
