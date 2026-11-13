// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.bc -emit-llvm %s -o - | FileCheck %s
// REQUIRES: amdgpu-registered-target

void runtime(unsigned long long ub, unsigned long long step) {
#pragma omp target teams distribute parallel for schedule(static, 1) thread_limit(256)
  for (unsigned long long i = 0; i < ub; i += step)
    __builtin_trap();
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_runtime_l{{[0-9]+}}_omp_outlined(
// CHECK: [[DIST:%.*]] = sub i64 {{%.*}}, 0
// CHECK-NEXT: [[SPAN:%.*]] = sub i64 [[DIST]], 1
// CHECK-NEXT: [[STEP:%.*]] = load i64,
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], [[STEP]]
// CHECK-NEXT: [[COUNT:%.*]] = add i64 [[LAST]], 1
// CHECK-NEXT: [[BOUND:%.*]] = sub i64 [[COUNT]], 1
// CHECK-NEXT: store i64 [[BOUND]],
// CHECK: [[PRECOND:%.*]] = icmp ult i64 0,
// CHECK-NEXT: br i1 [[PRECOND]], label %omp.precond.then, label %omp.precond.end
// CHECK: call void @__kmpc_parallel_60(

void constant(void) {
#pragma omp target teams distribute parallel for schedule(static, 1) thread_limit(256)
  for (unsigned long long i = 0; i < 0xFFFFFFFFFFFFFFFFULL;
       i += 0x1111111111111111ULL)
    __builtin_trap();
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.*}}_constant_l{{[0-9]+}}_omp_outlined(
// CHECK: store i64 14, ptr %.omp.comb.ub.ascast
// CHECK: icmp ult i64 {{%.*}}, 15
// CHECK: call void @__kmpc_parallel_60(
