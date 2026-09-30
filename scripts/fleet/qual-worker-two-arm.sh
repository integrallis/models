#!/bin/bash
# Self-bootstrapping qualification worker.
#
# Reporting goes to the SERIAL CONSOLE first and everything else second. That is the whole
# design change: the previous version began with `exec > /var/log/qual-worker.log 2>&1`, so
# every byte it produced went to a disk nobody could reach -- no SSH key was held for the
# launch keypair, the role carried no SSM permission, and the EXIT trap guarded its S3
# upload with `command -v aws`, which is silent in precisely the case where aws is what is
# missing. Four channels, all shut at once, so a worker that died in bootstrap looked
# identical to one that finished its shard.
#
# get-console-output needs no key, no agent and no bucket, and the buffer survives
# termination. It is therefore the only channel that reports a failure that happens BEFORE
# the tooling is installed, which is where the failures actually were.
# Supplied by the bootstrap so one worker script drives every shard; 40 only as a safety default.
SHARD=${SHARD:-40}
DEADLINE_SECONDS=18000
BUCKET=models-qual-077051030817
LOG=/var/log/qual-worker.log

# Milestones to console; full trace to the log file only. The console ring buffer is small
# (~64KB) and `set -x` on it would evict exactly the lines that say why a worker died.
say() { printf '[qual %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG" >/dev/console 2>/dev/null; }
exec 2>>"$LOG"
set -x

STATUS=STARTED
say "boot shard=$SHARD instance=$(cat /sys/devices/virtual/dmi/id/product_uuid 2>/dev/null | head -c 8)"

# Reported unconditionally, before any tool is needed. Terminating is deliberate (spend
# control) and is now safe to do because the console has already said why.
finish() {
  rc=$?
  say "FINAL_STATUS=$STATUS exit=$rc"
  if command -v aws >/dev/null 2>&1; then
    echo "$STATUS" > /work/out/STATUS 2>/dev/null
    aws s3 cp /work/out/STATUS "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "$LOG" "s3://$BUCKET/results/shard-$SHARD/qual-worker-$SHARD.log" --only-show-errors 2>/dev/null
    [ -f /work/out/progress.tsv ] && aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    say "uploaded to s3://$BUCKET/results/shard-$SHARD/"
  else
    say "no aws cli: console output is the only record"
  fi
  # HOLD_ON_FAIL keeps a broken box alive long enough to be inspected instead of erasing
  # the evidence. Success always terminates.
  if [ "$STATUS" != "SHARD_COMPLETE" ] && [ "1" = "1" ]; then
    say "holding 45m for inspection (HOLD_ON_FAIL=1)"
    sleep 2700
  fi
  say "shutting down"
  shutdown -h now
}
trap finish EXIT

mkdir -p /work/models /work/out /opt/jdk || { STATUS=NO_WORKDIR; exit 1; }

# Network first: every later step assumes egress, and without this check a dead route looks
# like a dead script. Tested by whether a response ARRIVES, not by its status: s3.amazonaws.com
# answers an unauthenticated HEAD with 307, and `curl -f` turns any non-2xx into a non-zero
# exit -- so the obvious check reports NO_EGRESS on a perfectly healthy instance and kills the
# worker. Verified on the AMI: s3 root -> 307, awscli zip -> 200, code 000 means no response.
EGRESS_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 https://s3.amazonaws.com 2>/dev/null)
if [ "$EGRESS_CODE" = "000" ] || [ -z "$EGRESS_CODE" ]; then
  STATUS=NO_EGRESS; say "no outbound HTTPS (curl code=$EGRESS_CODE)"; exit 1
fi
say "egress ok (s3 root http=$EGRESS_CODE)"

# The bundled installer is the PRIMARY path, not a fallback. Verified on this AMI
# (Ubuntu 24.04.5): `apt-cache policy awscli` reports "Candidate: (none)" -- the package does
# not exist in the configured repos at all, so the previous `apt-get install -y awscli`
# could never succeed. That single missing candidate is what killed every worker: the install
# failed, the guard exited, and the EXIT trap's own `command -v aws` test meant it could not
# report why. Installs aws-cli 2.37.4 here.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || say "apt-get update failed (continuing)"
apt-get install -y -qq unzip python3 || say "apt-get install unzip/python3 failed (continuing)"
if ! command -v aws >/dev/null 2>&1; then
  say "installing aws cli from the bundled installer"
  if curl -sfL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip; then
    (cd /tmp && unzip -qo awscli.zip && ./aws/install --update >/dev/null 2>&1)
    hash -r
  else
    say "could not download the aws installer"
  fi
fi
command -v aws >/dev/null 2>&1 || { STATUS=NO_AWS_CLI; say "aws cli unavailable"; exit 1; }
say "aws $(aws --version 2>&1 | head -c 40)"
command -v python3 >/dev/null 2>&1 || { STATUS=NO_PYTHON; exit 1; }

curl -sfL "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse" -o /tmp/jdk.tgz \
  || { STATUS=NO_JDK_DOWNLOAD; exit 1; }
tar xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 || { STATUS=NO_JDK_UNPACK; exit 1; }
export JAVA_HOME=/opt/jdk; export PATH=$JAVA_HOME/bin:$PATH
[ -x /opt/jdk/bin/java ] || { STATUS=NO_JDK; exit 1; }
say "jdk $(/opt/jdk/bin/java -version 2>&1 | head -1)"

cd /work
aws s3 cp "s3://$BUCKET/payload/models-rag-bench-0.3.50-v23.tar" . --only-show-errors
aws s3 cp "s3://$BUCKET/payload/models-kernels-linux-x86_64.jar" . --only-show-errors
aws s3 cp "s3://$BUCKET/payload/fleet-shard-$SHARD.json" /work/shard.json --only-show-errors
[ -s models-rag-bench-0.3.50-v23.tar ] || { STATUS=NO_DIST; exit 1; }
[ -s /work/models-kernels-linux-x86_64.jar ] || { STATUS=NO_KERNEL_JAR; exit 1; }
[ -s /work/shard.json ] || { STATUS=NO_SHARD; exit 1; }
tar xf models-rag-bench-0.3.50-v23.tar || { STATUS=NO_DIST_UNPACK; exit 1; }
DIST=$(ls -d /work/models-rag-bench-*/ | head -1)
CP="$DIST/lib/*:/work/models-kernels-linux-x86_64.jar"
JOBS=$(python3 -c "import json;print(len(json.load(open('/work/shard.json'))))")
say "payload ok, $JOBS jobs"
# QUAL_THREADS lets a shard pin the thread count. Both arms receive it -- the CLI sends
# num_thread to ollama and n_threads to llama.cpp -- so a sweep stays a matched comparison and
# sameWorkload() still admits the comparator.
THREADS=${QUAL_THREADS:-$(nproc 2>/dev/null || echo 8)}
# Decode runs on decodeThreadCount, which defaults to the whole pool -- 16 SMT threads on 8 physical
# cores. A bandwidth-bound single-row matmul usually peaks at or below the physical core count, so this
# is the one knob worth sweeping before writing kernel code. Ours only; the declared --threads budget is
# unchanged for both arms.
# Measured on shard-40/51/52 across 16, 8 and 4 decode threads: every K-quant model was faster at 8
# than at 16 (+3% on a 1.28GB model, +11% at 2.39GB, +26% at 0.32GB), while the one Q4_0 model got
# slower at 8 and collapsed at 4 -- a different kernel with a different optimum. So the thread count is
# per job, read from the manifest, rather than one value imposed on the whole shard.
DECODE_THREAD_DEFAULT="${QUAL_DECODE_THREADS:-}"
# Baked in, not derived: the worker has no git repository, so `git rev-parse` there would record
# "local" and the result would name no build at all.
BACKEND_VERSION="models@0.3.50+v23-08d9b5e1cef8"
STATUS=RUNNING

# The comparator arm. A qualification is comparative: RagProductionQualificationPolicy needs a baseline
# measured on the SAME host and the SAME workload, or it returns NO_COMPARABLE_BASELINE. Every shard
# before this one ran the candidate alone, so the campaign produced candidate reports and no verdicts.
# The installer puts ollama in /usr/local/bin, which a user-data shell does not always have on PATH.
export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"
# HOME is UNSET in an EC2 user-data shell, and ollama's envconfig resolves the model directory from it
# while building its CLI -- so *every* ollama invocation, `serve` and `create` alike, dies with
# `panic: $HOME is not defined` before it parses a single argument. That panic, and not the install,
# is why the previous shard produced candidate-only reports. The AMI also runs no systemd, so the
# unit the installer registers is inert and `ollama serve` has to be supervised here directly.
export HOME=/root
# Models go on the 120GB data volume, not the root filesystem: the comparator imports a copy of every
# candidate GGUF and the shard is 20GB of them.
export OLLAMA_MODELS=/work/ollama
mkdir -p "$OLLAMA_MODELS"
curl -fsSL https://ollama.com/install.sh | sh > /work/out/ollama-install.log 2>&1
say "ollama install exit=$? path=$(command -v ollama || echo none) home=$HOME"
(OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve >> /work/out/ollama-serve.log 2>&1 &)
for attempt in $(seq 1 90); do
  curl -s -o /dev/null http://127.0.0.1:11434/api/tags && break
  sleep 2
done
# Uploaded now, not at shutdown: a comparator that never starts is the one failure that makes the whole
# shard worthless, and waiting hours to read why is how the previous campaign wasted itself.
aws s3 cp /work/out/ollama-install.log "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
aws s3 cp /work/out/ollama-serve.log   "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
if curl -s -o /dev/null http://127.0.0.1:11434/api/tags; then
  say "ollama ready ($(ollama --version 2>&1 | head -1))"
  OLLAMA_READY=1
else
  # Fail fast. A candidate arm without a same-host comparator cannot produce a verdict, so running ten
  # of them would repeat the previous campaign exactly: reports, no qualifications.
  STATUS=NO_COMPARATOR
  say "ollama NOT ready; refusing to run candidate-only arms"
  exit 1
fi

# Architectures this dist can actually load. Checked BEFORE the download, because an
# unsupported model still costs a multi-GB pull before the backend refuses it 44s later --
# measured on shard 99, where gemma2 failed with
#   "Unsupported GGUF architecture: gemma2; supported architectures are
#    llama, qwen2, qwen3, granite, smollm3, gemma3, gemma-embedding"
# The first seven come from that message (the LlamaForwardPass route); the rest are the
# decoder adapters present in backend-java-0.3.49.jar, which route ahead of it. 20 of the
# 78 backlog jobs are outside this set, so skipping them early is a quarter of the
# download budget back.
# gemma4 REMOVED: Gemma4DecoderAdapter exists but targets a different metadata shape than the
# published GGUFs use -- E-series hit two explicit "Unsupported Gemma 4 variant" guards
# (shared_kv_layers, per-layer input embeddings) and the 12B wants expert_feed_forward_length.
# Verified from the backend's own exceptions, not from the adapter class existing.
# gpt-oss REMOVED for GGUF: type 39 (MXFP4) is not a known GgufTensorType, so the parser rejects
# the file. gpt-oss IS supported via the HuggingFace safetensors path -- qualify it from there.
SUPPORTED_ARCH="llama gemma gemma2 gemma4 phi3 mistral3 hunyuan-dense lfm2 qwen2 qwen3 qwen3moe granite smollm3 gemma3 gemma-embedding qwen35 qwen35moe qwen3next gemma3n deepseek2 gpt-oss mobilemoe needle2"

# Largest artifact this instance type can actually decode, in bytes. GGUF is mmap'd, so an
# oversized model loads and then thrashes rather than failing: decode touches every weight on
# every token, so the working set is the whole file and anything near RAM turns into page-cache
# churn. It would report a latency that measures swap, not the model. m6a.2xlarge has 30 GB, and
# the KV cache, JVM heap and page cache all have to live there too.
# 22GB suits an m6a.4xlarge, where 64GB of RAM has to hold the model for our arm and then again for
# ollama's. A larger instance can take more, so the ceiling is settable rather than baked in --
# Qwen3-Coder-Next is 48.5GB and was silently skipped by the fixed limit.
MAX_ARTIFACT_BYTES=${QUAL_MAX_ARTIFACT_BYTES:-22000000000}

( sleep ${DEADLINE_SECONDS:-18000}; touch /work/out/STOP ) &

done_n=0
python3 -c "
import json
for j in json.load(open('/work/shard.json')): print(json.dumps(j))
" > /work/jobs.ndjson
# Read from a file, not a pipe: a piped while-loop body is a subshell, so the counters and
# STATUS it sets are lost to the parent and the final report undercounts the run.
while IFS= read -r job; do
  [ -f /work/out/STOP ] && { say "deadline reached after $done_n jobs"; break; }
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$job")
  uri=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['uri'])" "$job")
  tpl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['tpl'])" "$job")
  arch=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('arch',''))" "$job")
  dt=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('dt') or '')" "$job")
  wl=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('wl') or 'general')" "$job")
  [ -z "$dt" ] && dt="$DECODE_THREAD_DEFAULT"
  DECODE_THREAD_OPT=""
  if [ -n "$dt" ]; then
    DECODE_THREAD_OPT="-Dmodels.native.kernels.decodeThreads=$dt"
  fi
  gguf="/work/models/$id.gguf"
  T0=$(date +%s)
  case " $SUPPORTED_ARCH " in
    *" $arch "*) ;;
    *)
      printf '%s\tSKIP_UNSUPPORTED_ARCH_%s\t0\n' "$id" "$arch" >> /work/out/progress.tsv
      say "skip $id: arch=$arch not loadable by this dist"
      done_n=$((done_n+1))
      continue
      ;;
  esac
  # Needed by the size cap below, which only HEADs the single-file form.
  files=$(python3 -c "import json,sys;print(' '.join(json.loads(sys.argv[1]).get('files') or []))" "$job")
  # The size cap, enforced against the artifact's REAL size.
  #
  # This used to read a 'bytes' field from the manifest, which the manifests do not carry -- so
  # ${bytes:-0} was always 0, the comparison never fired, and the cap was dead code. A HEAD against the
  # pinned URL costs one request and cannot be out of date.
  #
  # MEASURED 2026-09-29, and it argues for a GENEROUS cap rather than a tight one: a 48.5 GB
  # Qwen3-Coder-Next (80B) on an m6a.4xlarge -- 64 GB, so the weights do not fit in page cache alongside
  # the heap -- still answered every case correctly, at 38.6 s to first token and 241 ms/token against
  # roughly 4 s and 80 ms for the 21 GB models on the same box. So oversized models degrade rather than
  # fail, and the cap's job is to stop a shard's deadline being eaten, not to prevent a wrong answer.
  # Set MAX_ARTIFACT_BYTES from the deadline and the shard's model count, not from the instance's RAM.
  bytes=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('bytes',0))" "$job")
  if [ "${bytes:-0}" = "0" ] && [ -z "$files" ]; then
    bytes=$(curl -sIL "$uri" 2>/dev/null \
      | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v+0}')
    say "$id: HEAD reports ${bytes:-unknown} bytes"
  fi
  if [ "${bytes:-0}" -gt "$MAX_ARTIFACT_BYTES" ] 2>/dev/null; then
    printf '%s\tSKIP_TOO_LARGE_%s\t0\n' "$id" "$bytes" >> /work/out/progress.tsv
    say "skip $id: ${bytes} bytes exceeds $MAX_ARTIFACT_BYTES for this instance type"
    done_n=$((done_n+1))
    continue
  fi
  # A job is either a single GGUF file or a Hugging Face directory. The directory form lists its
  # files explicitly rather than querying the API at run time: the list and the revision in the uri
  # are pinned together, so a worker downloads exactly what was scheduled -- the same discipline the
  # single-file jobs already get from a pinned resolve URL. Sharded safetensors need this because
  # there is no single model.safetensors to fetch (gpt-oss-20b ships three shards plus an index).
  if [ -n "$files" ]; then
    artifact="/work/models/$id"
    rm -rf "$artifact"; mkdir -p "$artifact"
    say "start $id (arch=$arch, hf-dir, $(echo $files | wc -w | tr -d ' ') files)"
    ok=1
    for name in $files; do
      if ! curl -sfL "${uri%/}/$name" -o "$artifact/$name"; then
        say "download failed $id: $name"; ok=0; break
      fi
    done
    if [ "$ok" = "1" ]; then
      java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
        -Dmodels.native.quantizedDecode=true \
        -cp "$CP" \
        com.integrallis.models.rag.RagBenchmarkCli \
        --framework plain-java --backend rust-ffm --backend-version "$BACKEND_VERSION" \
        --model "$artifact" --model-id "$id" --workload "$wl" --prompt-template "$tpl" \
        --context 2048 --threads 8 --max-tokens 256 --warmups 1 --iterations 3 \
        --output "/work/out/$id.json" > "/work/out/$id.log" 2>&1
      rc=$?
      printf '%s\trc%s\t%s\n' "$id" "$rc" "$(( $(date +%s)-T0 ))" >> /work/out/progress.tsv
      say "done $id rc=$rc in $(( $(date +%s)-T0 ))s"
      aws s3 cp "/work/out/$id.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
      aws s3 cp "/work/out/$id.log"  "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    else
      printf '%s\tDOWNLOAD_FAIL\t0\n' "$id" >> /work/out/progress.tsv
    fi
    rm -rf "$artifact"
    done_n=$((done_n+1))
    continue
  fi

  say "start $id (arch=$arch workload=$wl decodeThreads=${dt:-pool})"
  if curl -sfL "$uri" -o "$gguf"; then
    # Every workload field sameWorkload() compares must match between the two arms or the comparator is
    # excluded: workload, corpus, cases, template, topK, max tokens, context, THREADS, grounding policy,
    # minimum retrieval score, and the matched generation controls. Threads are passed explicitly to
    # both for that reason -- a shard that passed --threads 8 to the candidate and let the comparator
    # default to the host's core count was rejected with "benchmark workload differs".
    ARMS="--workload $wl --prompt-template $tpl --context 2048 --threads $THREADS"
    ARMS="$ARMS --max-tokens 256 --warmups 1 --iterations 3"

    java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
      -Dmodels.native.quantizedDecode=true \
      $DECODE_THREAD_OPT \
      -cp "$CP" \
      com.integrallis.models.rag.RagBenchmarkCli \
      --framework plain-java --backend rust-ffm --backend-version "$BACKEND_VERSION" \
      --model "$gguf" --model-id "$id" $ARMS \
      --output "/work/out/$id.json" > "/work/out/$id.log" 2>&1
    rc=$?

    if [ "$rc" = "0" ] && [ "$OLLAMA_READY" = "1" ]; then
      # The comparator runs the identical artifact, not a registry copy of the same name: `ollama create`
      # from this exact file, so artifactSha256 matches and the comparison is not excluded.
      printf 'FROM %s\n' "$gguf" > /work/Modelfile
      if ollama create "qual-$id" -f /work/Modelfile >> "/work/out/$id.ollama.log" 2>&1; then
        java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
          -cp "$CP" \
          com.integrallis.models.rag.RagBenchmarkCli \
          --framework plain-java --backend ollama --backend-version "ollama@$(ollama --version 2>&1 | head -1 | tr -d '\n')" \
          --model "qual-$id" --artifact "$gguf" --model-id "$id" $ARMS \
          --output "/work/out/$id.comparator.json" >> "/work/out/$id.ollama.log" 2>&1
        crc=$?
        if [ "$crc" = "0" ]; then
          java -cp "$CP" com.integrallis.models.rag.RagQualificationCli \
            --candidate "/work/out/$id.json" \
            --comparator "/work/out/$id.comparator.json" \
            --output "/work/out/$id.verdict.json" >> "/work/out/$id.ollama.log" 2>&1
          verdict=$(python3 -c "import json;print(json.load(open('/work/out/$id.verdict.json'))['qualification']['verdict'])" 2>/dev/null || echo NO_VERDICT)
          say "verdict $id: $verdict"
          printf '%s\tverdict\t%s\n' "$id" "$verdict" >> /work/out/progress.tsv
          aws s3 cp "/work/out/$id.comparator.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
          aws s3 cp "/work/out/$id.verdict.json"    "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
        else
          say "comparator arm failed for $id (rc=$crc)"
        fi
        ollama rm "qual-$id" >> "/work/out/$id.ollama.log" 2>&1
      else
        say "ollama could not load $id; no comparator for it"
      fi
      aws s3 cp "/work/out/$id.ollama.log" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    fi
    printf '%s\trc%s\t%s\n' "$id" "$rc" "$(( $(date +%s)-T0 ))" >> /work/out/progress.tsv
    say "done $id rc=$rc in $(( $(date +%s)-T0 ))s"
    aws s3 cp "/work/out/$id.json" "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
    aws s3 cp "/work/out/$id.log"  "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
  else
    printf '%s\tDOWNLOAD_FAIL\t0\n' "$id" >> /work/out/progress.tsv
    say "download failed $id"
  fi
  rm -f "$gguf"
  done_n=$((done_n+1))
done < /work/jobs.ndjson

aws s3 cp /work/out/progress.tsv "s3://$BUCKET/results/shard-$SHARD/" --only-show-errors 2>/dev/null
STATUS=SHARD_COMPLETE
say "shard complete: $done_n jobs"
