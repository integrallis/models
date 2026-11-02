export HOME=/root
quiet() { for i in $(seq 1 120); do n=$(ps -eo args | grep -cE '[R]agBenchmarkCli|[l]lama-server|[l]lama-bench' || true); [ "$n" -eq 0 ] && { sleep 3; return 0; }; sleep 10; done; }
echo "=== CPU / memory ==="
lscpu | grep -E "^Model name|^Socket|^Core|^Thread|^NUMA node\(s\)|avx2|avx512" | head -6
grep -oE "avx2|avx512f" /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "=== STREAM-like triad (sum over 2GB, 16 threads) ==="
cat > /root/bw.c <<'EOC'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
int main(){ size_t n=1UL<<28; /* 256M floats = 1GB */
  float *a=aligned_alloc(64,n*4); memset(a,1,n*4);
  double best=0;
  for(int r=0;r<5;r++){ double t0=omp_get_wtime(); double s=0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(size_t i=0;i<n;i++) s+=a[i];
    double t=omp_get_wtime()-t0; double gbs=(n*4.0)/t/1e9;
    if(gbs>best)best=gbs; printf("  read pass %d: %.1f GB/s (sum=%.0f)\n",r,gbs,s); }
  printf("  BEST READ BANDWIDTH: %.1f GB/s\n",best); return 0; }
EOC
gcc -O3 -march=native -fopenmp -o /root/bw /root/bw.c 2>&1 | head -3
quiet; OMP_NUM_THREADS=16 /root/bw
echo "=== roofline signature: tok/s x bytes across two quants ==="
for m in /root/mathstral/mathstral-7B-v0.1-Q4_K_S.gguf /root/mathstral-km/m.gguf; do
  quiet
  sz=$(stat -c %s "$m")
  line=$(/root/llama.cpp/b/bin/llama-bench -m "$m" -p 0 -n 128 -t 16 -r 3 2>/dev/null | grep tg128)
  tps=$(echo "$line" | awk -F'|' '{print $(NF-1)}' | awk '{print $1}')
  echo "  $(basename $m) bytes=$sz tg128=$tps  => $(python3 -c "print(f'{$tps*$sz/1e9:.1f} GB/s')")"
done
