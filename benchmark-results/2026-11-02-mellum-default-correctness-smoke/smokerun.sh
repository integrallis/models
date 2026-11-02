set -e
export HOME=/root
. /root/.cargo/env 2>/dev/null || true
SO=/root/bench/crate/target/release/libjmodels_kernels.so
CP=$(ls /root/bench/lib/*.jar | tr '\n' ':')
MODEL=/root/mellum/m.gguf
MODEL_ID=mellum2_12b_a2_5b_thinking_q4_k_m
MODEL_SHA=$(sha256sum "$MODEL" | awk '{print $1}')
# Mirrors scripts/run-controlled-rag-qualification.sh: its COMMON_ARGS, its --warmups 0
# --iterations 1 default-correctness phase, its jq predicate and its record shape. MODELS_VERSION
# follows its own format, naming the commit the library was built from.
MODELS_VERSION="models@a991ae40bec1bafeef69864067791d7218776fbb com.integrallis:vectors-core@0.1.32"
THREADS=$(nproc)
OUT=/root/smoke/default-correctness
mkdir -p $OUT
REPORT=$OUT/models-rust-ffm.json

# the script's own hygiene: no competing inference, and a cold file cache
n=$(ps -eo args | grep -cE '[R]agBenchmarkCli|[l]lama-server|[o]llama runner' || true)
[ "$n" -eq 0 ] || { echo "competing inference processes present: $n"; exit 1; }
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "(could not drop caches)"

/root/jdk/bin/java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -XX:ActiveProcessorCount=$THREADS -Dmodels.native.kernels.library=$SO -cp "$CP" \
  com.integrallis.models.rag.RagBenchmarkCli \
  --backend rust-ffm --backend-version "$MODELS_VERSION" \
  --model "$MODEL" --output "$REPORT" \
  --framework plain-java --model-id "$MODEL_ID" --workload coding \
  --prompt-template chatml-no-think --context 2048 --threads $THREADS \
  --top-k 1 --max-tokens 64 \
  --warmups 0 --iterations 1 > $OUT/run.log 2>&1 || { echo "CLI rc=$?"; tail -20 $OUT/run.log; exit 1; }

echo "### the script's predicate, verbatim"
if jq -e \
  --arg backend rust-ffm --arg model_id "$MODEL_ID" --arg artifact_sha "$MODEL_SHA" \
  '.backend == $backend
   and .modelId == $model_id
   and .artifactSha256 == $artifact_sha
   and .settings.warmups == 0
   and .settings.iterations == 1
   and .settings.generationControls.promptCache == "longest-common-prefix"
   and .summary.totalAttempts > 0
   and .summary.successfulAttempts == .summary.totalAttempts
   and .summary.correctAnswerRate == 1
   and .summary.abstentionAccuracy == 1
   and (.failures | length) == 0' "$REPORT" >/dev/null; then
  echo "  PREDICATE PASS"
else
  echo "  PREDICATE FAIL"
  jq '{backend,modelId,artifactSha256,settings:{warmups:.settings.warmups,iterations:.settings.iterations,promptCache:.settings.generationControls.promptCache},summary:{totalAttempts:.summary.totalAttempts,successfulAttempts:.summary.successfulAttempts,correctAnswerRate:.summary.correctAnswerRate,abstentionAccuracy:.summary.abstentionAccuracy},failures:(.failures|length)}' "$REPORT"
  exit 1
fi
if jq -e '.backendDiagnostics.environment["native-quantized-decode"] == "true"' "$REPORT" >/dev/null; then
  echo "  SHIM ROUTE PASS (native-quantized-decode == true)"
else
  echo "  SHIM ROUTE FAIL"; exit 1
fi
REPORT_SHA=$(sha256sum "$REPORT" | awk '{print $1}')
jq -n --arg backend rust-ffm --arg model_id "$MODEL_ID" --arg artifact_sha "$MODEL_SHA" \
  --arg report "default-correctness/models-rust-ffm.json" --arg report_sha "$REPORT_SHA" \
  --argjson total_attempts "$(jq '.summary.totalAttempts' "$REPORT")" \
  '{schemaVersion: 1,
    kind: "model-backend-default-correctness-smoke",
    configuration: "library-defaults",
    backend: $backend,
    modelId: $model_id,
    artifactSha256: $artifact_sha,
    report: $report,
    reportSha256: $report_sha,
    totalAttempts: $total_attempts,
    successfulAttempts: $total_attempts,
    tuningSystemProperties: []}' > /root/smoke/smoke.json
echo "### smoke.json"; cat /root/smoke/smoke.json
echo "### decode/prefill for the record"
jq '{decode:.summary.p50DecodeTokensPerSecond, prefill:.summary.p50PrefillTokensPerSecond}' "$REPORT"
echo "### SMOKE DONE"
