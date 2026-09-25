// Achievable read bandwidth on this host, at the thread counts the kernel uses.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

static char *buf; static size_t len; static int nthreads;
static void *work(void *arg) {
  long id = (long)arg;
  size_t chunk = len / nthreads;
  const uint64_t *p = (const uint64_t *)(buf + id * chunk);
  size_t n = chunk / 8;
  uint64_t acc = 0;
  for (size_t i = 0; i < n; i += 8) {
    acc += p[i] + p[i+1] + p[i+2] + p[i+3] + p[i+4] + p[i+5] + p[i+6] + p[i+7];
  }
  return (void *)(uintptr_t)acc;
}
int main(int argc, char **argv) {
  len = 3UL << 30;              // 3 GiB, comfortably past any cache
  buf = aligned_alloc(64, len);
  memset(buf, 1, len);
  for (int t = 1; t <= 8; t *= 2) {
    nthreads = t;
    double best = 1e9;
    for (int round = 0; round < 3; round++) {
      struct timespec a, b;
      pthread_t th[8];
      clock_gettime(CLOCK_MONOTONIC, &a);
      for (long i = 0; i < t; i++) pthread_create(&th[i], NULL, work, (void *)i);
      for (long i = 0; i < t; i++) pthread_join(th[i], NULL);
      clock_gettime(CLOCK_MONOTONIC, &b);
      double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
      if (s < best) best = s;
    }
    printf("  %d thread%s: %.1f GB/s\n", t, t == 1 ? " " : "s", len / best / 1e9);
  }
  return 0;
}
