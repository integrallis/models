// Head-to-head of ggml's per-row Q4_K x Q8_K vec_dot against our Rust kernel, in one process, on
// identical weights, SINGLE-THREADED and interleaved. Single thread removes thread scaling and
// memory-bandwidth saturation as confounds, so what is left is kernel efficiency.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"

#define QK_K 256
#define Q4K_BLOCK_BYTES 144
#define Q5K_BLOCK_BYTES 176

extern void *jmodels_kernels_context_create(unsigned int thread_count);
extern int jmodels_kernels_context_destroy(void *ctx);
extern int jmodels_quantized_f32_batched_matmul_with_context(
    const void *ctx, unsigned int format, const unsigned char *weights, unsigned long long weight_bytes,
    const float *input, unsigned long long input_elements, float *output, unsigned long long output_elements,
    unsigned int batch_size, unsigned int rows, unsigned int cols);

static double now_s(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

static unsigned int rng = 20261101u;
static unsigned int xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(int argc, char **argv) {
  ggml_cpu_init();
  int cols = 4096;
  int rows = argc > 1 ? atoi(argv[1]) : 4096;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  const char *tname = argc > 3 ? argv[3] : "q4_K";
  int is_q5 = strcmp(tname, "q5_K") == 0;
  enum ggml_type gt = is_q5 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
  unsigned int our_format = is_q5 ? 4u : 2u;
  int block_bytes = is_q5 ? Q5K_BLOCK_BYTES : Q4K_BLOCK_BYTES;
  int blocks = cols / QK_K;
  size_t row_bytes = (size_t)blocks * block_bytes;
  size_t bytes = (size_t)rows * row_bytes;

  unsigned char *w = aligned_alloc(64, (bytes + 63) & ~63UL);
  // Valid Q4_K blocks: d/dmin as sane fp16, 6-bit scale fields, random nibbles.
  ggml_fp16_t d = ggml_fp32_to_fp16(0.0125f), dmin = ggml_fp32_to_fp16(0.0021f);
  for (size_t off = 0; off < bytes; off += block_bytes) {
    unsigned char *b = w + off;
    for (int i = 0; i < block_bytes; i++) b[i] = (unsigned char)(xr() & 0xFF);
    memcpy(b + 0, &d, 2); memcpy(b + 2, &dmin, 2);
    for (int i = 4; i < 16; i++) b[i] &= 0x3F;
  }
  float *in = malloc(cols * sizeof(float));
  for (int i = 0; i < cols; i++) in[i] = (float)((int)(xr() % 2000) - 1000) / 500.0f;

  float *out_g = malloc(rows * sizeof(float));
  float *out_o = malloc(rows * sizeof(float));

  // ggml needs the activation pre-quantized to Q8_K; ours quantizes internally. That is O(cols)
  // against O(rows*cols) of dot work, so it is negligible here and is noted, not corrected.
  const struct ggml_type_traits_cpu *tc = ggml_get_type_traits_cpu(gt);
  const struct ggml_type_traits_cpu *tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
  size_t q8_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
  void *q8 = aligned_alloc(64, (q8_bytes + 63) & ~63UL);
  tq->from_float(in, q8, cols);

  void *ctx = jmodels_kernels_context_create(1);
  if (!ctx) { printf("context_create failed\n"); return 1; }

  printf("type=%s rows=%d cols=%d weights=%.2f MiB  (L3 on this box is 12 MiB)\n",
         tname, rows, cols, bytes / 1048576.0);

  double best_g = 0, best_o = 0;
  for (int r = 0; r < reps; r++) {
    double t0 = now_s();
    for (int i = 0; i < rows; i++)
      tc->vec_dot(cols, &out_g[i], 0, w + (size_t)i * row_bytes, 0, q8, 0, 1);
    double tg = now_s() - t0;

    double t1 = now_s();
    int st = jmodels_quantized_f32_batched_matmul_with_context(
        ctx, our_format, w, bytes, in, cols, out_o, rows, 1, rows, cols);
    double to = now_s() - t1;
    if (st != 0) { printf("ours returned status %d\n", st); return 1; }

    double gg = bytes / tg / 1e9, go = bytes / to / 1e9;
    if (gg > best_g) best_g = gg;
    if (go > best_o) best_o = go;
    printf("  rep %d  ggml %6.2f GB/s   ours %6.2f GB/s   ours/ggml %.3f\n", r, gg, go, go / gg);
  }

  printf("  ggml out[0..4]: %g %g %g %g %g\n", out_g[0], out_g[1], out_g[2], out_g[3], out_g[4]);
  printf("  ours out[0..4]: %g %g %g %g %g\n", out_o[0], out_o[1], out_o[2], out_o[3], out_o[4]);
  double md = 0, mx = 0;
  for (int i = 0; i < rows; i++) {
    double diff = fabs((double)out_g[i] - (double)out_o[i]);
    double mag = fabs((double)out_g[i]) + 1e-6;
    md += diff / mag; if (diff / mag > mx) mx = diff / mag;
  }
  printf("BEST   ggml %6.2f GB/s   ours %6.2f GB/s   ours/ggml %.3f   ggml/ours %.3fx\n",
         best_g, best_o, best_o / best_g, best_g / best_o);
  printf("agreement: mean rel diff %.3e  max %.3e  (sanity: both computed the same dot)\n",
         md / rows, mx);
  jmodels_kernels_context_destroy(ctx);
  return 0;
}
