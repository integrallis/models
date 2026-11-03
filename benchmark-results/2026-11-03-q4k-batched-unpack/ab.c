// A/B harness: drives jmodels_quantized_f32_batched_matmul_with_context on Q4_K (format 2),
// the exact route mellum's prefill takes, and reports wall time plus an FNV-1a hash of the
// output so a codegen change that alters arithmetic cannot hide behind a timing number.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

#define QK_K 256
#define Q4K_BLOCK 144

typedef void* (*ctx_create_fn)(uint32_t);
typedef void  (*ctx_destroy_fn)(void*);
typedef int32_t (*matmul_fn)(const void*, uint32_t, const uint8_t*, uint64_t,
                             const float*, uint64_t, float*, uint64_t,
                             uint32_t, uint32_t, uint32_t);

static uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    const char* lib = argv[1];
    uint32_t rows = (uint32_t)atoi(argv[2]);
    uint32_t cols = (uint32_t)atoi(argv[3]);
    uint32_t batch = (uint32_t)atoi(argv[4]);
    int reps = atoi(argv[5]);
    uint32_t threads = (uint32_t)atoi(argv[6]);

    void* h = dlopen(lib, RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    ctx_create_fn ctx_create = (ctx_create_fn)dlsym(h, "jmodels_kernels_context_create");
    ctx_destroy_fn ctx_destroy = (ctx_destroy_fn)dlsym(h, "jmodels_kernels_context_destroy");
    matmul_fn matmul = (matmul_fn)dlsym(h, "jmodels_quantized_f32_batched_matmul_with_context");
    if (!ctx_create || !matmul) { fprintf(stderr, "dlsym failed\n"); return 1; }

    uint64_t blocks_per_row = cols / QK_K;
    uint64_t weight_bytes = (uint64_t)rows * blocks_per_row * Q4K_BLOCK;
    uint8_t* weights = malloc(weight_bytes);
    for (uint64_t i = 0; i < weight_bytes; i++) weights[i] = (uint8_t)(rnd() >> 7);
    // Keep the fp16 scale/min fields in a sane exponent range so sums stay finite.
    for (uint64_t b = 0; b < (uint64_t)rows * blocks_per_row; b++) {
        uint8_t* blk = weights + b * Q4K_BLOCK;
        blk[1] = (uint8_t)(0x2c | (blk[1] & 0x03));
        blk[3] = (uint8_t)(0x2c | (blk[3] & 0x03));
    }
    uint64_t in_n = (uint64_t)batch * cols, out_n = (uint64_t)batch * rows;
    float* in = malloc(in_n * sizeof(float));
    float* out = malloc(out_n * sizeof(float));
    for (uint64_t i = 0; i < in_n; i++) in[i] = ((float)(rnd() % 2000) - 1000.0f) / 1000.0f;

    void* ctx = ctx_create(threads);
    int32_t rc = matmul(ctx, 2, weights, weight_bytes, in, in_n, out, out_n, batch, rows, cols);
    if (rc != 0) { fprintf(stderr, "matmul rc=%d\n", rc); return 1; }
    uint64_t hash = fnv1a(out, out_n * sizeof(float));

    for (int w = 0; w < 3; w++) matmul(ctx, 2, weights, weight_bytes, in, in_n, out, out_n, batch, rows, cols);
    double best = 1e300, total = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        matmul(ctx, 2, weights, weight_bytes, in, in_n, out, out_n, batch, rows, cols);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
        total += dt;
        if (fnv1a(out, out_n * sizeof(float)) != hash) { fprintf(stderr, "OUTPUT CHANGED between reps\n"); return 2; }
    }
    double gflop = 2.0 * rows * cols * batch / 1e9;
    printf("hash=%016llx best_ms=%.3f mean_ms=%.3f best_gflops=%.2f\n",
           (unsigned long long)hash, best * 1e3, total / reps * 1e3, gflop / best);
    if (ctx_destroy) ctx_destroy(ctx);
    return 0;
}
