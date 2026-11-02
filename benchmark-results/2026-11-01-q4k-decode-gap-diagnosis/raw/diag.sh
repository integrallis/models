export HOME=/root
set -e
mkdir -p /root/diag
CP=$(ls /root/models-rag-bench/lib/*.jar | tr '\n' ':')
cat > /root/diag/prof.jfc <<'EOX'
<?xml version="1.0" encoding="UTF-8"?>
<configuration version="2.0">
  <event name="jdk.ExecutionSample"><setting name="enabled">true</setting><setting name="period">1 ms</setting></event>
  <event name="jdk.NativeMethodSample"><setting name="enabled">true</setting><setting name="period">1 ms</setting></event>
  <event name="jdk.ThreadPark"><setting name="enabled">true</setting><setting name="threshold">1 ms</setting></event>
</configuration>
EOX
echo "### candidate arm: mathstral Q4_K_S, rust-ffm, 16 threads"
/root/jdk/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -Dmodels.native.kernels.library=/root/model-kernels/target/release/libjmodels_kernels.so \
  -XX:StartFlightRecording=settings=/root/diag/prof.jfc,filename=/root/diag/decode.jfr \
  -cp "$CP" com.integrallis.models.rag.RagBenchmarkCli \
  --framework plain-java --backend rust-ffm --backend-version diag@local \
  --model /root/mathstral/mathstral-7B-v0.1-Q4_K_S.gguf --model-id mathstral-diag \
  --workload general --prompt-template zephyr --top-k 3 --context 2048 \
  --threads 16 --max-tokens 256 --warmups 1 --iterations 3 \
  --output /root/diag/candidate.json > /root/diag/candidate.log 2>&1 || echo "### rc=$? (see log)"
echo "### tail of log"; tail -5 /root/diag/candidate.log
echo "### decode metrics"
python3 - <<'EOP'
import json
d=json.load(open('/root/diag/candidate.json'))
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)): walk(v,p+'/'+k)
            elif any(s in k.lower() for s in ('decode','token','latency','throughput','prefill')):
                print(f"  {p}/{k} = {v}")
    elif isinstance(o,list):
        for i,v in enumerate(o[:2]): walk(v,p+f'[{i}]')
walk(d)
EOP
