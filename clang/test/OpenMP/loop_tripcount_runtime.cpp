// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -O1 -Wno-pass-failed -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SEMA
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-enable-irbuilder -O1 -Wno-pass-failed -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,IRBUILDER

// Keep bounds and steps dynamic, including signed ranges wider than the signed
// counter type. The division must precede the addition of the first iteration.
extern "C" void body();
#define DIRECTIVE "omp for"
#define LOOP(NAME, TYPE, CMP, UPDATE)                  \
  extern "C" void NAME(TYPE lb, TYPE ub, TYPE step) { \
    _Pragma(DIRECTIVE)                                \
    for (TYPE i = lb; i CMP ub; UPDATE)                 \
      body();                                         \
  }

// CHECK-LABEL: define {{.*}} @unsigned_up(
// CHECK: icmp ult i64 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i64 %lb, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i64 %ub, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(unsigned_up, unsigned long long, <, i += step)
// CHECK-LABEL: define {{.*}} @unsigned_down(
// CHECK: icmp ugt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i64 %ub, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i64 %lb, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(unsigned_down, unsigned long long, >, i -= step)
// CHECK-LABEL: define {{.*}} @unsigned_inclusive_up(
// CHECK: icmp ugt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[SPAN:%.*]] = sub nuw i64 %ub, %lb
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(unsigned_inclusive_up, unsigned long long, <=, i += step)
// CHECK-LABEL: define {{.*}} @unsigned_inclusive_down(
// CHECK: icmp ult i64 %lb, %ub
// CHECK: br i1
// CHECK: [[SPAN:%.*]] = sub nuw i64 %lb, %ub
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(unsigned_inclusive_down, unsigned long long, >=, i -= step)
// CHECK-LABEL: define {{.*}} @signed_up(
// CHECK: icmp slt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i64 %lb, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i64 %ub, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(signed_up, long long, <, i += step)
// CHECK-LABEL: define {{.*}} @signed_down(
// CHECK: icmp sgt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i64 %ub, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i64 %lb, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(signed_down, long long, >, i -= step)
// CHECK-LABEL: define {{.*}} @signed_inclusive_up(
// CHECK: icmp sgt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[SPAN:%.*]] = sub i64 %ub, %lb
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(signed_inclusive_up, long long, <=, i += step)
// CHECK-LABEL: define {{.*}} @signed_inclusive_down(
// CHECK: icmp slt i64 %lb, %ub
// CHECK: br i1
// CHECK: [[SPAN:%.*]] = sub i64 %lb, %ub
// CHECK-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i64 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(signed_inclusive_down, long long, >=, i -= step)
// CHECK-LABEL: define {{.*}} @unsigned_32(
// CHECK: icmp ult i32 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i32 %lb, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i32 %ub, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i32 [[SPAN]], %step
// SEMA: store i32 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i32 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(unsigned_32, unsigned, <, i += step)
// CHECK-LABEL: define {{.*}} @signed_32(
// CHECK: icmp slt i32 %lb, %ub
// CHECK: br i1
// CHECK: [[NOTLOW:%.*]] = xor i32 %lb, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i32 %ub, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i32 [[SPAN]], %step
// SEMA: store i32 [[LAST]], ptr %.omp.ub
// IRBUILDER: phi i32 [ [[LAST]], %{{.*}} ], [ -1, %{{.*}} ]
LOOP(signed_32, int, <, i += step)
// IRBuilder worksharing supports only 32- and 64-bit counters. SIMD exercises
// the same distance closure for the other widths without a runtime ABI call.
#undef DIRECTIVE
#define DIRECTIVE "omp simd"
// CHECK-LABEL: define {{.*}} @unsigned_16(
// SEMA: [[LAST:%.*]] = udiv i32
// SEMA-NEXT: {{%.*}} = add i32 [[LAST]], 1
// IRBUILDER: [[SPAN:%.*]] = add nsw i32 {{%.*}}, -1
// IRBUILDER: [[LAST:%.*]] = sdiv i32 [[SPAN]],
LOOP(unsigned_16, unsigned short, <, i += step)
// CHECK-LABEL: define {{.*}} @signed_16(
// SEMA: [[LAST:%.*]] = udiv i32
// SEMA-NEXT: {{%.*}} = add i32 [[LAST]], 1
// IRBUILDER: [[SPAN:%.*]] = add nsw i32 {{%.*}}, -1
// IRBUILDER: [[LAST:%.*]] = sdiv i32 [[SPAN]],
LOOP(signed_16, short, >, i -= step)
// CHECK-LABEL: define {{.*}} @unsigned_8(
// SEMA: [[LAST:%.*]] = udiv i32
// SEMA-NEXT: {{%.*}} = add i32 [[LAST]], 1
// IRBUILDER: [[SPAN:%.*]] = add nsw i32 {{%.*}}, -1
// IRBUILDER: [[LAST:%.*]] = sdiv i32 [[SPAN]],
LOOP(unsigned_8, unsigned char, >, i -= step)
// CHECK-LABEL: define {{.*}} @signed_8(
// SEMA: [[LAST:%.*]] = udiv i32
// SEMA-NEXT: {{%.*}} = add i32 [[LAST]], 1
// IRBUILDER: [[SPAN:%.*]] = add nsw i32 {{%.*}}, -1
// IRBUILDER: [[LAST:%.*]] = sdiv i32 [[SPAN]],
LOOP(signed_8, signed char, <, i += step)
// CHECK-LABEL: define {{.*}} @unsigned_128(
// CHECK: [[NOTLOW:%.*]] = xor i128 %lb, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i128 %ub, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i128 [[SPAN]], %step
// CHECK-NEXT: {{%.*}} = add i128 [[LAST]], 1
LOOP(unsigned_128, unsigned __int128, <, i += step)
// CHECK-LABEL: define {{.*}} @signed_128(
// CHECK: [[NOTLOW:%.*]] = xor i128 %ub, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i128 %lb, [[NOTLOW]]
// CHECK-NEXT: [[LAST:%.*]] = udiv i128 [[SPAN]], %step
// CHECK-NEXT: {{%.*}} = add i128 [[LAST]], 1
LOOP(signed_128, __int128, >, i -= step)

// The pointer subtraction must stay a pointer difference; only the subsequent
// division/rounding changes. Its result must already fit ptrdiff_t.
// CHECK-LABEL: define {{.*}} @pointers(
// CHECK: icmp ult ptr %lb, %ub
// CHECK-DAG: ptrtoaddr ptr %ub to i64
// CHECK-DAG: ptrtoaddr ptr %lb to i64
// CHECK: [[SPAN:%.*]] = add i64
// SEMA-NEXT: [[LAST:%.*]] = sdiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
extern "C" void pointers(char *lb, char *ub, __PTRDIFF_TYPE__ step) {
#pragma omp for
  for (char *i = lb; i < ub; i += step)
    body();
}

// A random access iterator's subtraction may return a signed difference near
// its maximum, so adding the step before division can overflow there too.
struct Iterator {
  long long pos;
  Iterator &operator+=(long long n) { pos += n; return *this; }
  Iterator operator+(long long n) const { return {pos + n}; }
  long long operator-(Iterator other) const { return pos - other.pos; }
  bool operator<(Iterator other) const { return pos < other.pos; }
};
// CHECK-LABEL: define {{.*}} @iterators(
// CHECK: [[NOTLOW:%.*]] = xor i64 %lb.coerce, -1
// CHECK-NEXT: [[SPAN:%.*]] = add i64 %ub.coerce, [[NOTLOW]]
// SEMA-NEXT: [[LAST:%.*]] = sdiv i64 [[SPAN]], %step
// SEMA: store i64 [[LAST]], ptr %.omp.ub
// IRBUILDER-NEXT: [[LAST:%.*]] = udiv i64 [[SPAN]], %step
extern "C" void iterators(Iterator lb, Iterator ub, long long step) {
#pragma omp for
  for (Iterator i = lb; i < ub; i += step)
    body();
}

// The signed difference fits, and one increment reaches the endpoint, but
// adding the step to the difference would overflow.
// CHECK-LABEL: define {{.*}} @iterator_boundary(
// CHECK: call void @body()
// CHECK-NEXT: ret void
extern "C" void iterator_boundary() {
#pragma omp simd
  for (Iterator i = {0}; i < Iterator{9223372036854775807LL};
       i += 9223372036854775807LL)
    body();
}
