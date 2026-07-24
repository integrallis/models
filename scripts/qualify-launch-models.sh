#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qualify-launch-models.sh \
  --catalog PATH --queue PATH --runner PATH --output-dir PATH [options]

Options:
  --models-dir PATH       Artifact cache (default: $HOME/.jvllm/models)
  --llama-server PATH     llama-server executable (default: llama-server)
  --models-revision SHA   Models source revision recorded with backend version
  --llama-version TEXT    llama.cpp build identifier (default: unknown)
  --target COUNT          Stop after COUNT production-qualified models
  --port PORT             Local llama-server port (default: 8080)
  --dry-run               Validate and print the campaign without downloading
EOF
}

catalog=
queue=
runner=
output_dir=
models_dir="${HOME}/.jvllm/models"
llama_server=llama-server
models_revision=unknown
llama_version=unknown
target=
port=8080
dry_run=false

while (($#)); do
  case "$1" in
    --catalog) catalog=$2; shift 2 ;;
    --queue) queue=$2; shift 2 ;;
    --runner) runner=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --models-dir) models_dir=$2; shift 2 ;;
    --llama-server) llama_server=$2; shift 2 ;;
    --models-revision) models_revision=$2; shift 2 ;;
    --llama-version) llama_version=$2; shift 2 ;;
    --target) target=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in catalog queue runner output_dir; do
  if [[ -z ${!required} ]]; then
    echo "--${required//_/-} is required" >&2
    usage >&2
    exit 2
  fi
done

for command in jq curl sha256sum; do
  command -v "$command" >/dev/null || {
    echo "Required command is unavailable: $command" >&2
    exit 2
  }
done

[[ -f $catalog ]] || { echo "Catalog not found: $catalog" >&2; exit 2; }
[[ -f $queue ]] || { echo "Queue not found: $queue" >&2; exit 2; }
if [[ $dry_run == false ]]; then
  [[ -x $runner ]] || { echo "RAG runner is not executable: $runner" >&2; exit 2; }
  command -v "$llama_server" >/dev/null || {
    echo "llama-server is unavailable: $llama_server" >&2
    exit 2
  }
fi

if [[ -z $target ]]; then
  target=$(jq -er '.targetQualifiedModels' "$queue")
fi
[[ $target =~ ^[1-9][0-9]*$ ]] || { echo "--target must be positive" >&2; exit 2; }

mkdir -p "$models_dir" "$output_dir/logs"
summary="$output_dir/qualification-summary.tsv"
printf 'model_id\tstatus\tperformance_tier\treport\n' >"$summary"

server_pid=
cleanup_server() {
  if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap cleanup_server EXIT INT TERM

qualified=0
attempted=0
validated=0
while IFS= read -r model_id; do
  row=$(jq -ec --arg id "$model_id" '.models[] | select(.id == $id)' "$catalog") || {
    echo "Catalog entry not found: $model_id" >&2
    printf '%s\tcatalog-error\t-\t-\n' "$model_id" >>"$summary"
    continue
  }
  count=$(jq -r --arg id "$model_id" '[.models[] | select(.id == $id)] | length' "$catalog")
  if [[ $count != 1 ]]; then
    echo "Catalog ID is not unique: $model_id" >&2
    printf '%s\tcatalog-error\t-\t-\n' "$model_id" >>"$summary"
    continue
  fi

  supports_chat=$(jq -r '(.capabilities | index("chat")) != null' <<<"$row")
  supports_llama=$(jq -r '.backends["llama.cpp"] == true' <<<"$row")
  if [[ $supports_chat != true || $supports_llama != true ]]; then
    echo "Skipping $model_id: chat or llama.cpp support is absent" >&2
    printf '%s\tunsupported\t-\t-\n' "$model_id" >>"$summary"
    continue
  fi

  name=$(jq -r '.name' <<<"$row")
  uri=$(jq -r '.downloadUri' <<<"$row")
  sha256=$(jq -r '.sha256' <<<"$row")
  size_bytes=$(jq -r '.sizeBytes' <<<"$row")
  license=$(jq -r '.license' <<<"$row")
  catalog_path=$(jq -r '.localPath' <<<"$row")
  file_name=$(basename "${catalog_path//\$\{user.home\}\//$HOME/}")
  artifact="$models_dir/$file_name"
  report="$output_dir/${model_id}-llama-native.json"
  validated=$((validated + 1))

  printf '%s\t%s\t%s bytes\t%s\t%s\n' \
    "$model_id" "$name" "$size_bytes" "$license" "$file_name"
  if [[ $dry_run == true ]]; then
    continue
  fi

  attempted=$((attempted + 1))
  if [[ ! -f $artifact ]] || [[ $(stat -c '%s' "$artifact") != "$size_bytes" ]]; then
    partial="${artifact}.partial"
    curl -fL --retry 5 --retry-delay 2 --continue-at - --output "$partial" "$uri"
    mv "$partial" "$artifact"
  fi
  actual_sha=$(sha256sum "$artifact" | awk '{print $1}')
  if [[ $actual_sha != "$sha256" ]]; then
    echo "SHA-256 mismatch for $model_id: $actual_sha != $sha256" >&2
    printf '%s\tchecksum-failed\t-\t-\n' "$model_id" >>"$summary"
    continue
  fi

  cleanup_server
  log="$output_dir/logs/${model_id}-llama-server.log"
  "$llama_server" \
    -m "$artifact" \
    --host 127.0.0.1 \
    --port "$port" \
    -c 2048 \
    -t 8 \
    --jinja >"$log" 2>&1 &
  server_pid=$!

  ready=false
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ $ready != true ]]; then
    echo "llama-server failed to start for $model_id; see $log" >&2
    printf '%s\tserver-failed\t-\t-\n' "$model_id" >>"$summary"
    cleanup_server
    continue
  fi

  if "$runner" \
    --framework plain-java \
    --backend llama.cpp \
    --backend-version "${llama_version}+models.${models_revision}" \
    --model "$file_name" \
    --model-id "$model_id" \
    --artifact "$artifact" \
    --endpoint "http://127.0.0.1:${port}" \
    --prompt-template native-chat \
    --context 2048 \
    --threads 8 \
    --pid "$server_pid" \
    --top-k 1 \
    --max-tokens 64 \
    --warmups 1 \
    --iterations 3 \
    --output "$report"; then
    tier=$(jq -r '.performanceTier' "$report")
    plan=$(jq -r '.backendDiagnostics.planVersion' "$report")
    successful=$(jq -r '.summary.successfulAttempts == .summary.totalAttempts' "$report")
    if [[ $plan == local-http-v1 && $successful == true ]] \
      && [[ $tier == PRODUCTION_READY || $tier == USABLE ]]; then
      qualified=$((qualified + 1))
      printf '%s\tqualified\t%s\t%s\n' "$model_id" "$tier" "$report" >>"$summary"
    else
      printf '%s\trejected\t%s\t%s\n' "$model_id" "$tier" "$report" >>"$summary"
    fi
  else
    printf '%s\tbenchmark-failed\t-\t%s\n' "$model_id" "$report" >>"$summary"
  fi
  cleanup_server

  echo "Campaign progress: $qualified qualified / $attempted attempted (target $target)"
  if ((qualified >= target)); then
    break
  fi
done < <(jq -er '.modelIds[]' "$queue")

if [[ $dry_run == true ]]; then
  echo "Qualification campaign validated: $validated runnable catalog entries"
  if ((validated < target)); then
    exit 1
  fi
  exit 0
fi

echo "Qualification campaign complete: $qualified qualified / $attempted attempted"
echo "Summary: $summary"
if ((qualified < target)); then
  exit 1
fi
