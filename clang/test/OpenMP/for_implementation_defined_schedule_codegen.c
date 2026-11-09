// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-optzns -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-optzns -o - %s | FileCheck %s

void unspecified(int n, int *out) {
// CHECK-LABEL: define{{.*}} @unspecified(
// CHECK: call void @__kmpc_for_static_init_4{{u?}}({{.*}}), !omp.implementation_defined ![[IMPL:[0-9]+]]
#pragma omp for
  for (int i = 0; i < n; ++i)
    out[i] = i;
}

void specified(int n, int *out) {
// CHECK-LABEL: define{{.*}} @specified(
// CHECK: call void @__kmpc_for_static_init_4{{u?}}(
// CHECK-NOT: !omp.implementation_defined
// CHECK: call void @__kmpc_for_static_fini(
#pragma omp for schedule(static)
  for (int i = 0; i < n; ++i)
    out[i] = i;
}

// CHECK: ![[IMPL]] = !{}
