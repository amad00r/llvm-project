// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

#include <stdio.h>

// Volatile inputs force the device to compute the count at runtime. In
// particular, (ub - 1 + step) / step used to wrap and produce zero.
#define CHECK_LOOP(TYPE, UTYPE, START, CMP, STOP, UPDATE)                                                         \
  do {                                                                                                            \
    volatile TYPE start = (START), stop = (STOP);                                                                 \
    volatile TYPE step = (TYPE)(((UTYPE)-1) / 15);                                                                \
    int count = 0;                                                                                                \
    _Pragma("omp target teams distribute parallel for schedule(static, 1) thread_limit(256) map(tofrom : count)") \
    for (TYPE i = start; i CMP stop; UPDATE) {                                                                    \
      _Pragma("omp atomic update")                                                                                \
      ++count;                                                                                                    \
    }                                                                                                             \
    if (count != 15)                                                                                              \
      return 1;                                                                                                   \
  } while (0)

int main(void) {
  // Exercise promotions as well as the 32- and 64-bit difference types.
#define CHECK_TYPE(T, U, LO, HI)                                              \
  CHECK_LOOP(T, U, LO, <, HI, i += step);                                     \
  CHECK_LOOP(T, U, HI, >, LO, i -= step);                                     \
  CHECK_LOOP(T, U, LO, <=, (HI) - 1, i += step);                              \
  CHECK_LOOP(T, U, HI, >=, (LO) + 1, i -= step)
  CHECK_TYPE(signed char, unsigned char, -128, 127);
  CHECK_TYPE(unsigned char, unsigned char, 0, 255);
  CHECK_TYPE(short, unsigned short, -32768, 32767);
  CHECK_TYPE(unsigned short, unsigned short, 0, 65535);
  CHECK_TYPE(int, unsigned int, (-2147483647 - 1), 2147483647);
  CHECK_TYPE(unsigned int, unsigned int, 0, ~0U);
  CHECK_TYPE(long long, unsigned long long, (-9223372036854775807LL - 1),
             9223372036854775807LL);
  CHECK_TYPE(unsigned long long, unsigned long long, 0, ~0ULL);

  volatile unsigned long long step = 0x1111111111111111ULL;
  volatile unsigned long long ub = ~0ULL;
  int count = 0;
#pragma omp target teams distribute parallel for schedule(static, 1) thread_limit(256) map(tofrom : count)
  for (unsigned long long i = 0; i < ub; i += step) {
#pragma omp atomic update
    ++count;
  }
  if (count != 15)
    return 2;

  count = 0;
#pragma omp target teams distribute parallel for collapse(2) map(tofrom : count)
  for (unsigned long long i = 0; i < ~0ULL; i += 0x1111111111111111ULL)
    for (int j = 0; j < 2; ++j) {
#pragma omp atomic update
      ++count;
    }
  if (count != 30)
    return 3;

  count = 0;
  volatile unsigned long long empty_ub = 0;
#pragma omp target teams distribute parallel for map(tofrom : count)
  for (unsigned long long i = ub; i < empty_ub; i += step) {
#pragma omp atomic update
    ++count;
  }
  if (count != 0)
    return 4;
  puts("PASS");
  // CHECK: PASS
  return 0;
}
