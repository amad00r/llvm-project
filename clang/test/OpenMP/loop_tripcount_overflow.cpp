// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp-simd -O1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -O1 -emit-llvm %s -o - | FileCheck %s

// Each loop has 15 iterations, including when the range crosses the signed
// boundary. The old round-up numerator wrapped for the wider types.
#define COUNT(NAME, TYPE, START, CMP, STOP, UPDATE) \
  extern "C" unsigned NAME() {                     \
    unsigned count = 0;                            \
    _Pragma("omp simd reduction(+:count)")          \
    for (TYPE i = START; i CMP STOP; UPDATE)         \
      ++count;                                     \
    return count;                                  \
  }

#define BOUNDARIES(NAME, TYPE, MIN, MAX, STEP)                         \
  COUNT(NAME##_lt, TYPE, MIN, <, MAX, i += STEP)                       \
  COUNT(NAME##_le, TYPE, MIN, <=, (MAX) - 1, i += STEP)                \
  COUNT(NAME##_gt, TYPE, MAX, >, MIN, i -= STEP)                       \
  COUNT(NAME##_ge, TYPE, MAX, >=, (MIN) + 1, i -= STEP)

using u128 = unsigned __int128;
constexpr u128 U128Max = (u128)-1;
constexpr __int128 S128Max = U128Max >> 1;

// CHECK-LABEL: define {{.*}} @u8_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u8_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u8_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u8_ge(
// CHECK: ret i32 15
BOUNDARIES(u8, unsigned char, 0, 255, 17)

// CHECK-LABEL: define {{.*}} @s8_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s8_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s8_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s8_ge(
// CHECK: ret i32 15
BOUNDARIES(s8, signed char, -128, 127, 17)

// CHECK-LABEL: define {{.*}} @u16_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u16_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u16_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u16_ge(
// CHECK: ret i32 15
BOUNDARIES(u16, unsigned short, 0, 65535, 4369)

// CHECK-LABEL: define {{.*}} @s16_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s16_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s16_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s16_ge(
// CHECK: ret i32 15
BOUNDARIES(s16, short, -32768, 32767, 4369)

// CHECK-LABEL: define {{.*}} @u32_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u32_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u32_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u32_ge(
// CHECK: ret i32 15
BOUNDARIES(u32, unsigned, 0, ~0U, 0x11111111U)

// CHECK-LABEL: define {{.*}} @s32_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s32_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s32_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s32_ge(
// CHECK: ret i32 15
BOUNDARIES(s32, int, (-2147483647 - 1), 2147483647, 0x11111111)

// CHECK-LABEL: define {{.*}} @u64_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u64_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u64_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u64_ge(
// CHECK: ret i32 15
BOUNDARIES(u64, unsigned long long, 0, ~0ULL, 0x1111111111111111ULL)

// CHECK-LABEL: define {{.*}} @s64_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s64_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s64_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s64_ge(
// CHECK: ret i32 15
BOUNDARIES(s64, long long, (-9223372036854775807LL - 1),
           9223372036854775807LL, 0x1111111111111111LL)

// CHECK-LABEL: define {{.*}} @u128_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u128_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u128_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @u128_ge(
// CHECK: ret i32 15
BOUNDARIES(u128, u128, 0, U128Max, U128Max / 15)

// CHECK-LABEL: define {{.*}} @s128_lt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s128_le(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s128_gt(
// CHECK: ret i32 15
// CHECK-LABEL: define {{.*}} @s128_ge(
// CHECK: ret i32 15
BOUNDARIES(s128, __int128, (-S128Max - 1), S128Max, (__int128)(U128Max / 15))

// Unit steps do not need to round up a strict range. Check the inclusive
// boundary and retain the existing behavior for an unrepresentable count.
// CHECK-LABEL: define {{.*}} @unit_up(
// CHECK: ret i32 15
COUNT(unit_up, unsigned long long, ~0ULL - 15, <, ~0ULL, ++i)
// CHECK-LABEL: define {{.*}} @unit_down(
// CHECK: ret i32 15
COUNT(unit_down, long long, (-9223372036854775807LL - 1) + 15, >,
      (-9223372036854775807LL - 1), --i)
// A wide step can overflow even when the loop variable is promoted from an
// 8- or 16-bit type. Its conversion back to the counter reaches the endpoint.
// CHECK-LABEL: define {{.*}} @wide_step_8(
// CHECK: ret i32 1
COUNT(wide_step_8, unsigned char, 0, <, 255, i += ~0U)
// CHECK-LABEL: define {{.*}} @wide_step_16(
// CHECK: ret i32 1
COUNT(wide_step_16, unsigned short, 0, <, 65535, i += ~0U)
// CHECK-LABEL: define {{.*}} @full_range(
// CHECK: ret i32 0
COUNT(full_range, unsigned long long, 0, <=, ~0ULL, ++i)

// Empty loops must still be skipped, even if computing their range wraps.
// CHECK-LABEL: define {{.*}} @empty_equal(
// CHECK: ret i32 0
COUNT(empty_equal, unsigned long long, ~0ULL, <, ~0ULL, i += 17)
// CHECK-LABEL: define {{.*}} @empty_reversed(
// CHECK: ret i32 0
COUNT(empty_reversed, long long, 9223372036854775807LL, <,
      (-9223372036854775807LL - 1), i += 17)

// Each dimension must be fixed before forming the collapsed product.
// CHECK-LABEL: define {{.*}} @collapsed(
// CHECK: ret i32 30
extern "C" unsigned collapsed() {
  unsigned count = 0;
#pragma omp simd collapse(2) reduction(+:count)
  for (unsigned long long i = 0; i < ~0ULL; i += 0x1111111111111111ULL)
    for (int j = 0; j < 2; ++j)
      ++count;
  return count;
}
