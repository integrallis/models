// How many int8 multiply-accumulates per cycle per core does this host actually reach,
// with AVX2 the way our Q4_K kernel does it, and with VNNI the way it could?
//
// Our kernel is AVX2-only. On Zen 3 that was the whole story, because Hetzner's dedicated line has
// no AVX-512 at all. MEASURED 2026-09-24 on a CCX33 (Milan), the Q4_K matmul reached 17.1
// MAC/cycle/core with weights streaming and 25.6 with them resident, and a forward pass ran at 15.6
// -- so ~79% of a decision is this arithmetic and it is not bandwidth bound (both arms used under a
// fifth of 38 GB/s).
//
// This measures the ceiling three ways on one host, one protocol, best of several rounds:
//
//   1. AVX2 the way we do it now: _mm256_maddubs_epi16 then _mm256_madd_epi16 to widen to int32.
//      Two instructions plus an add per 32 int8 products.
//   2. AVX2 + VNNI (_mm256_dpbusd_avx_epi32): one instruction per 32 int8 products, 256-bit.
//   3. AVX-512 + VNNI (_mm512_dpbusd_epi32): one instruction per 64 int8 products, 512-bit.
//
// The loops are deliberately register-resident and dependency-broken across several accumulators,
// so this is an instruction-throughput ceiling and NOT a claim about the matmul. A kernel also pays
// nibble unpacking, per-group scales and memory. The number to take away is the RATIO between arms:
// that is the most any rewrite of the inner loop could buy, and if the ratio is ~1 there is nothing
// to chase.
//
// Build:
//   gcc -O3 -march=native -o vnni vnni.c
//
// Reports absolute G-MAC/s on one core, NOT MAC/cycle. An earlier version of this file estimated the
// effective clock with a dependent-add chain; the compiler folded the chain into a single add and
// reported 411,395 GHz. Absolute throughput needs no clock estimate and cannot be wrong that way.
//
// It also pins `a` and `b` behind an empty asm barrier each iteration. Without that they are
// loop-invariant, the AVX2 arm's maddubs+madd hoists out of the loop entirely, and the arm measures
// a bare vector add -- which is how the first run of this file concluded that vpdpbusd was 0.64x
// SLOWER than doing more work.

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define ACCUMULATORS 8
#define INNER 1024
#define OUTER 20000

// Empty constraint that tells the compiler the register may have changed, so nothing that consumes
// it can be hoisted out of the loop. Costs no instructions.
#define KEEP(v) __asm__ __volatile__("" : "+x"(v))

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

__attribute__((target("avx2"))) static double bench_avx2(void) {
  __m256i a = _mm256_set1_epi8(3), b = _mm256_set1_epi8(5);
  __m256i ones = _mm256_set1_epi16(1);
  __m256i acc[ACCUMULATORS];
  for (int i = 0; i < ACCUMULATORS; i++) acc[i] = _mm256_setzero_si256();
  double best = 1e30;
  for (int round = 0; round < 3; round++) {
    double start = now();
    for (int outer = 0; outer < OUTER; outer++) {
      for (int inner = 0; inner < INNER / ACCUMULATORS; inner++) {
        for (int i = 0; i < ACCUMULATORS; i++) {
          KEEP(a);
          KEEP(b);
          // What our Q4_K inner loop does: unsigned*signed to int16, then widen to int32.
          __m256i products = _mm256_maddubs_epi16(a, b);
          acc[i] = _mm256_add_epi32(acc[i], _mm256_madd_epi16(products, ones));
        }
      }
    }
    double seconds = now() - start;
    if (seconds < best) best = seconds;
  }
  __m256i total = acc[0];
  for (int i = 1; i < ACCUMULATORS; i++) total = _mm256_add_epi32(total, acc[i]);
  volatile int sink = _mm256_extract_epi32(total, 0);
  (void)sink;
  // 32 int8 products per maddubs.
  double macs = (double)OUTER * INNER * 32;
  return macs / best / 1e9;
}

__attribute__((target("avx2,avxvnni"))) static double bench_avxvnni(void) {
  __m256i a = _mm256_set1_epi8(3), b = _mm256_set1_epi8(5);
  __m256i acc[ACCUMULATORS];
  for (int i = 0; i < ACCUMULATORS; i++) acc[i] = _mm256_setzero_si256();
  double best = 1e30;
  for (int round = 0; round < 3; round++) {
    double start = now();
    for (int outer = 0; outer < OUTER; outer++) {
      for (int inner = 0; inner < INNER / ACCUMULATORS; inner++) {
        for (int i = 0; i < ACCUMULATORS; i++) {
          KEEP(a);
          KEEP(b);
          acc[i] = _mm256_dpbusd_avx_epi32(acc[i], a, b);
        }
      }
    }
    double seconds = now() - start;
    if (seconds < best) best = seconds;
  }
  __m256i total = acc[0];
  for (int i = 1; i < ACCUMULATORS; i++) total = _mm256_add_epi32(total, acc[i]);
  volatile int sink = _mm256_extract_epi32(total, 0);
  (void)sink;
  double macs = (double)OUTER * INNER * 32;
  return macs / best / 1e9;
}

__attribute__((target("avx512f,avx512vnni"))) static double bench_avx512vnni(void) {
  __m512i a = _mm512_set1_epi8(3), b = _mm512_set1_epi8(5);
  __m512i acc[ACCUMULATORS];
  for (int i = 0; i < ACCUMULATORS; i++) acc[i] = _mm512_setzero_si512();
  double best = 1e30;
  for (int round = 0; round < 3; round++) {
    double start = now();
    for (int outer = 0; outer < OUTER; outer++) {
      for (int inner = 0; inner < INNER / ACCUMULATORS; inner++) {
        for (int i = 0; i < ACCUMULATORS; i++) {
          KEEP(a);
          KEEP(b);
          acc[i] = _mm512_dpbusd_epi32(acc[i], a, b);
        }
      }
    }
    double seconds = now() - start;
    if (seconds < best) best = seconds;
  }
  __m512i total = acc[0];
  for (int i = 1; i < ACCUMULATORS; i++) total = _mm512_add_epi32(total, acc[i]);
  volatile int sink = _mm512_reduce_add_epi32(total);
  (void)sink;
  // 64 int8 products per dpbusd at 512 bits.
  double macs = (double)OUTER * INNER * 64;
  return macs / best / 1e9;
}

int main(void) {
  double avx2 = bench_avx2();
  double avxvnni = bench_avxvnni();
  double avx512 = bench_avx512vnni();
  printf("  one core, int8 multiply-accumulate throughput\n\n");
  printf("  AVX2  maddubs+madd (what we do)  %7.1f G-MAC/s\n", avx2);
  printf("  AVX2  vpdpbusd                   %7.1f G-MAC/s   %.2fx\n", avxvnni, avxvnni / avx2);
  printf("  AVX512 vpdpbusd                  %7.1f G-MAC/s   %.2fx\n", avx512, avx512 / avx2);
  printf("\n  Ratio is the ceiling on an inner-loop rewrite, not a prediction for the matmul.\n");
  return 0;
}
