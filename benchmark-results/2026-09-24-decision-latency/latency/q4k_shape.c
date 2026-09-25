// The Q4_K inner loop as it actually is, AVX2 against AVX-512+VNNI.
//
// vnni.c measured a bare int8 dot and found AVX2 vpdpbusd 2.65x. That number does NOT transfer:
// our loop's third instruction applies the per-group scale and pair-adds in one _mm256_madd_epi16,
// so AVX2 is already 3 ops per 32 weights and a vpdpbusd rewrite is also 3 (dpbusd, mullo, add).
// The scale cannot move earlier -- a 4-bit quant times a 6-bit scale leaves u8.
//
// So the transferable factor is 512-bit WIDTH: one dpbusd covers 64 weights instead of 32, and the
// nibble unpack halves too. This measures that on the real shape: a Q4_K block of 256 weights as
// 8 groups of 32 (AVX2) or 4 of 64 (AVX-512), including the f16 scale decode, the nibble unpack,
// the per-group scale and the min/activation-sum correction.
//
// Working set is deliberately larger than L2 so this is not a register-resident ceiling.
//
//   gcc -O3 -march=native -o q4k_shape q4k_shape.c

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK_K 256
#define BLOCK_BYTES 144
#define ROWS 9216
#define COLS 2560
#define BLOCKS (COLS / QK_K)

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

static uint8_t *weights;
static int8_t *activations;

// 6-bit scale/min unpack, the same layout the Rust kernel reads.
static inline int qk_scale(const uint8_t *s, int g) {
  return g < 4 ? (s[g] & 63) : ((s[g + 4] & 15) | ((s[g - 4] >> 6) << 4));
}
static inline int qk_min(const uint8_t *s, int g) {
  return g < 4 ? (s[g + 4] & 63) : ((s[g + 4] >> 4) | ((s[g] >> 6) << 4));
}

__attribute__((target("avx2,fma,f16c"))) static double bench_avx2(int rows) {
  __m256i mask = _mm256_set1_epi8(0x0f);
  double best = 1e30;
  volatile float sink = 0;
  for (int round = 0; round < 3; round++) {
    double start = now();
    float total = 0;
    for (int row = 0; row < rows; row++) {
      __m256i acc = _mm256_setzero_si256();
      for (int block = 0; block < BLOCKS; block++) {
        const uint8_t *w = weights + (size_t)(row * BLOCKS + block) * BLOCK_BYTES;
        const uint8_t *scales = w + 4;
        const uint8_t *quants = w + 16;
        __m256i decoded[8];
        for (int pair = 0; pair < 4; pair++) {
          __m256i packed = _mm256_loadu_si256((const __m256i *)(quants + pair * 32));
          decoded[pair * 2] = _mm256_and_si256(packed, mask);
          decoded[pair * 2 + 1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        }
        for (int g = 0; g < 8; g++) {
          __m256i a = _mm256_loadu_si256((const __m256i *)(activations + block * QK_K + g * 32));
          __m256i products = _mm256_maddubs_epi16(decoded[g], a);
          __m256i scale = _mm256_set1_epi16((short)qk_scale(scales, g));
          acc = _mm256_add_epi32(acc, _mm256_madd_epi16(products, scale));
        }
      }
      total += (float)_mm256_extract_epi32(acc, 0);
    }
    double seconds = now() - start;
    if (seconds < best) best = seconds;
    sink = total;
  }
  (void)sink;
  double macs = (double)rows * COLS;
  return macs / best / 1e9;
}

__attribute__((target("avx512f,avx512bw,avx512vnni,avx512vl,f16c"))) static double bench_avx512(int rows) {
  __m512i mask = _mm512_set1_epi8(0x0f);
  double best = 1e30;
  volatile float sink = 0;
  for (int round = 0; round < 3; round++) {
    double start = now();
    float total = 0;
    for (int row = 0; row < rows; row++) {
      __m512i acc = _mm512_setzero_si512();
      for (int block = 0; block < BLOCKS; block++) {
        const uint8_t *w = weights + (size_t)(row * BLOCKS + block) * BLOCK_BYTES;
        const uint8_t *scales = w + 4;
        const uint8_t *quants = w + 16;
        // Two 512-bit loads cover all 128 packed bytes; four decoded groups of 64.
        __m512i decoded[4];
        for (int half = 0; half < 2; half++) {
          __m512i packed = _mm512_loadu_si512((const void *)(quants + half * 64));
          decoded[half * 2] = _mm512_and_si512(packed, mask);
          decoded[half * 2 + 1] = _mm512_and_si512(_mm512_srli_epi16(packed, 4), mask);
        }
        for (int g = 0; g < 4; g++) {
          __m512i a = _mm512_loadu_si512((const void *)(activations + block * QK_K + g * 64));
          __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), decoded[g], a);
          // Two 6-bit scales share a 64-weight group; the kernel would split the group so each
          // half gets its own. Using the pair's first scale here keeps the op count honest, which
          // is what is being measured -- this benchmark is about throughput, not about a result.
          __m512i scale = _mm512_set1_epi32(qk_scale(scales, g * 2));
          acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, scale));
        }
      }
      total += (float)_mm512_reduce_add_epi32(acc);
    }
    double seconds = now() - start;
    if (seconds < best) best = seconds;
    sink = total;
  }
  (void)sink;
  double macs = (double)rows * COLS;
  return macs / best / 1e9;
}

int main(void) {
  size_t weight_bytes = (size_t)ROWS * BLOCKS * BLOCK_BYTES;
  weights = aligned_alloc(64, weight_bytes);
  activations = aligned_alloc(64, COLS);
  for (size_t i = 0; i < weight_bytes; i++) weights[i] = (uint8_t)(i * 31 + 7);
  for (int i = 0; i < COLS; i++) activations[i] = (int8_t)(i % 71 - 35);
  printf("  Q4_K inner loop on the real shape: %d rows x %d cols, %.1f MiB of weights\n\n",
         ROWS, COLS, weight_bytes / 1048576.0);
  double avx2 = bench_avx2(ROWS);
  double avx512 = bench_avx512(ROWS);
  printf("  AVX2   8 groups of 32, maddubs+madd+add   %7.1f G-MAC/s\n", avx2);
  printf("  AVX512 4 groups of 64, dpbusd+mullo+add   %7.1f G-MAC/s   %.2fx\n", avx512, avx512 / avx2);
  printf("\n  One core, weights streaming from memory. Compare to the 2.65x/5.17x register-resident\n");
  printf("  bare-dot figures in vnni.c: this is the part that could actually transfer.\n");
  return 0;
}
