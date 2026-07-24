; RUN: opt --mtriple=amdgpu-amd-amdhsa -S -passes=openmp-kernel-versioning < %s | FileCheck %s --check-prefix=ENABLED
; RUN: opt --mtriple=amdgpu-amd-amdhsa -S -passes=openmp-kernel-versioning -openmp-kernel-versioning-disable < %s | FileCheck %s --check-prefix=DISABLED

define amdgpu_kernel void @kernel(i64 %num_work_items, i64 %total_sites) "kernel" {
entry:
  call void @kernel_omp_outlined(ptr null, ptr null, i64 %num_work_items,
                                 i64 %total_sites)
  ret void
}

define internal void @kernel_omp_outlined(ptr %tid, ptr %zero, i64 %num_work_items,
                                          i64 %total_sites) {
entry:
  %sum = add i64 %num_work_items, %total_sites
  call void @__kmpc_parallel_60(ptr null, i32 0, i32 0, i32 0, i32 0,
                                ptr @parallel_region, ptr null, ptr null,
                                i64 %sum, i32 0)
  ret void
}

define internal void @parallel_region(ptr %tid, ptr %bound_tid, i64 %lb, i64 %ub,
                                      i64 %num_work_items, i64 %total_sites) {
entry:
  %sum = add i64 %num_work_items, %total_sites
  ret void
}

declare void @__kmpc_parallel_60(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64,
                                 i32)

; ENABLED: define amdgpu_kernel void @kernel(i64 %num_work_items, i64 %total_sites) #0 {
; ENABLED: call void @kernel_omp_outlined(ptr null, ptr null, i64 800000000, i64 6250000)
; ENABLED: define internal void @kernel_omp_outlined(ptr %tid, ptr %zero, i64 %num_work_items, i64 %total_sites) {
; ENABLED: %sum = add i64 800000000, 6250000
; ENABLED: call void @__kmpc_parallel_60(ptr null, i32 0, i32 0, i32 0, i32 0, ptr @parallel_region, ptr null, ptr null, i64 %sum, i32 0)
; ENABLED: define internal void @parallel_region(ptr %tid, ptr %bound_tid, i64 %lb, i64 %ub, i64 %num_work_items, i64 %total_sites) {
; ENABLED: %sum = add i64 800000000, 6250000

; DISABLED: define amdgpu_kernel void @kernel(i64 %num_work_items, i64 %total_sites) #0 {
; DISABLED: call void @kernel_omp_outlined(ptr null, ptr null, i64 %num_work_items, i64 %total_sites)
; DISABLED: define internal void @kernel_omp_outlined(ptr %tid, ptr %zero, i64 %num_work_items, i64 %total_sites) {
; DISABLED: %sum = add i64 %num_work_items, %total_sites
; DISABLED: define internal void @parallel_region(ptr %tid, ptr %bound_tid, i64 %lb, i64 %ub, i64 %num_work_items, i64 %total_sites) {
; DISABLED: %sum = add i64 %num_work_items, %total_sites

!llvm.module.flags = !{!0, !1}

!0 = !{i32 7, !"openmp", i32 51}
!1 = !{i32 7, !"openmp-device", i32 51}
