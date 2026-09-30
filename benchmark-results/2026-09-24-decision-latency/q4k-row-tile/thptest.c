// Does this kernel actually give huge pages for an MADV_HUGEPAGE anonymous mapping?
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static long anon_huge_kb(void) {
  FILE *f = fopen("/proc/meminfo", "r");
  char key[64]; long value; long result = 0;
  while (fscanf(f, "%63s %ld kB\n", key, &value) >= 1) {
    if (strcmp(key, "AnonHugePages:") == 0) { result = value; break; }
  }
  fclose(f);
  return result;
}

int main(void) {
  size_t len = 2740937888UL;
  long before = anon_huge_kb();
  void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { perror("mmap"); return 1; }
  printf("  mapping at %p, 2MiB aligned: %s\n", p, ((unsigned long)p % (2UL<<20)) ? "no" : "yes");
  int rc = madvise(p, len, MADV_HUGEPAGE);
  printf("  madvise -> %d\n", rc);
  memset(p, 1, len);
  long after = anon_huge_kb();
  printf("  AnonHugePages %ld kB -> %ld kB  (delta %ld kB of %zu kB mapped)\n",
         before, after, after - before, len / 1024);
  return 0;
}
