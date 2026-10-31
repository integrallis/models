#!/usr/bin/env bash
# Disposable EC2 worker only. The launch manifest supplies an immutable payload and deadline.
set -euo pipefail
INPUT=${1:?manifest}
WORK=/work/released-pair
OUT=$WORK/results
mkdir -p "$OUT" "$WORK/jdk" "$WORK/models"
DEST=$(jq -er .resultsUri "$INPUT")
export AWS_DEFAULT_REGION=us-east-1
exec > >(tee -a "$OUT/worker.log" /dev/console) 2>&1
upload() { aws s3 cp "$OUT/" "$DEST/" --recursive --only-show-errors; }
cleanup() { rc=$?; trap - EXIT; printf '%s\n' "$rc" >"$OUT/worker.exit-code"; upload || true; shutdown -h now; exit "$rc"; }
trap cleanup EXIT
(while sleep 30; do upload || true; done) &
cp "$INPUT" "$OUT/manifest.json"
{ uname -a; lscpu; free -b; } >"$OUT/host.txt"
aws s3 cp "$(jq -er .payloadUri "$INPUT")" "$WORK/payload.tar.gz" --only-show-errors
printf '%s  %s\n' "$(jq -er .payloadSha256 "$INPUT")" "$WORK/payload.tar.gz" | sha256sum -c -
tar -xzf "$WORK/payload.tar.gz" -C "$WORK"
cp "$WORK/protocol.json" "$OUT/"
curl -fL --retry 3 --max-time 900 'https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.3%2B9/OpenJDK25U-jdk_x64_linux_hotspot_25.0.3_9.tar.gz' -o "$WORK/jdk.tar.gz"
printf '%s  %s\n' 69264a7a211bf5029830d07bc3370f879769d62ebc5b5488e90c9343a2da0e1f "$WORK/jdk.tar.gz" | sha256sum -c -
tar -xzf "$WORK/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1
SHARD=$(jq -er .shard "$INPUT")
cp "$WORK/shard-$SHARD.json" "$OUT/jobs.json"
if jq -e 'all(.[]; .comparator == "ollama")' "$OUT/jobs.json" >/dev/null; then
  export OLLAMA_MODELS=/work/ollama
  mkdir -p "$OLLAMA_MODELS"
  curl -fL --retry 3 --max-time 600 https://ollama.com/install.sh -o "$WORK/install-ollama.sh"
  OLLAMA_VERSION=0.40.13 sh "$WORK/install-ollama.sh" >"$OUT/ollama-install.log" 2>&1
  runuser -u root -- ollama serve >"$OUT/ollama-server.log" 2>&1 &
  for n in $(seq 1 60); do curl -fsS http://127.0.0.1:11434/api/version >"$OUT/ollama-version.json" && break; sleep 2; done
  jq -e '.version == "0.40.13"' "$OUT/ollama-version.json"
elif jq -e 'length == 1 and .[0].comparator == "prism-bonsai2-ptq1_0"' "$OUT/jobs.json" >/dev/null; then
  apt-get install -y -qq git build-essential cmake libcurl4-openssl-dev
  git clone --no-checkout https://github.com/PrismML-Eng/llama.cpp "$WORK/publisher"
  git -C "$WORK/publisher" checkout --detach e8fd209c975c80aa7fb36a8423571ff3b1b7e955
  [[ "$(git -C "$WORK/publisher" rev-parse HEAD)" == e8fd209c975c80aa7fb36a8423571ff3b1b7e955 ]]
  cmake -S "$WORK/publisher" -B "$WORK/publisher/build" -DGGML_CUDA=OFF -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >"$OUT/publisher-build.log" 2>&1
  cmake --build "$WORK/publisher/build" --target llama-server -j 8 >>"$OUT/publisher-build.log" 2>&1
  sha256sum "$WORK/publisher/build/bin/llama-server" >"$OUT/publisher-binary.sha256"
else
  echo 'unsupported mixed reference shard' >&2
  exit 2
fi
jq -c '.[]' "$OUT/jobs.json" >"$WORK/jobs.ndjson"
while IFS= read -r job; do
  id=$(jq -er .id <<<"$job")
  [[ "$id" =~ ^[a-z0-9_]+$ ]] || exit 2
  printf '%s\n' "$job" >"$WORK/job.json"
  model="$WORK/models/$id.gguf"
  curl -fL --retry 3 --max-time 1800 "$(jq -er .uri <<<"$job")" -o "$model"
  printf '%s  %s\n' "$(jq -er .sha256 <<<"$job")" "$model" | sha256sum -c -
  if jq -e '.smokeOnly == true' <<<"$job" >/dev/null; then
    python3 "$WORK/run-released-pair.py" --payload "$WORK/released" --job "$WORK/job.json" \
      --model "$model" --java "$WORK/jdk/bin/java" --output "$OUT/$id" --smoke-only
  elif [[ "$(jq -er .comparator <<<"$job")" == ollama ]]; then
    printf 'FROM %s\n' "$model" >"$OUT/$id.Modelfile"
    runuser -u root -- ollama create qualification -f "$OUT/$id.Modelfile" >"$OUT/$id.import.log" 2>&1
    python3 "$WORK/run-released-pair.py" --payload "$WORK/released" --job "$WORK/job.json" \
      --model "$model" --java "$WORK/jdk/bin/java" --output "$OUT/$id" \
      --reference-version ollama@0.40.13 --reference-model qualification
    runuser -u root -- ollama stop qualification
    runuser -u root -- ollama rm qualification
  else
    python3 "$WORK/run-released-qualification.py" --payload "$WORK/released" --java "$WORK/jdk/bin/java" \
      --model "$model" --reference-server "$WORK/publisher/build/bin/llama-server" \
      --reference-sha256 "$(cut -d ' ' -f 1 "$OUT/publisher-binary.sha256")" \
      --output "$OUT/$id" --result-uri "$DEST/$id"
  fi
  upload
  rm -- "$model"
done <"$WORK/jobs.ndjson"
printf '%s\n' complete >"$OUT/shard.complete"
