#!/usr/bin/env bash
# The throughput and latency measurement on a quiet dedicated box, both arms, against the RELEASED
# models library. Only public artifacts reach the box: Central jars (through jbang), the publisher's
# package and onnxruntime from PyPI/GitHub, the model from Hugging Face, and the scripts and corpora
# of this experiment. Run as root on a fresh Ubuntu 24.04 box after copying:
#   latency/ParadeeLatencyArm.java  latency/box-chain.sh  comparator/{comparator_arm.py,pins.py,
#   run_comparator_passes.sh,pyproject.toml,uv.lock}  corpus/*.json
# into /root/paradee/{latency,comparator,corpus}.
#
#   MODELS_VERSION=0.5.11 bash /root/paradee/latency/box-chain.sh
set -euo pipefail
MODELS_VERSION=${MODELS_VERSION:?set MODELS_VERSION to the released models version}
ROOT=/root/paradee
EV=$ROOT/evidence
log(){ echo "$(date -u +%FT%TZ) $*"; }
export DEBIAN_FRONTEND=noninteractive
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades >/dev/null 2>&1 || true
apt-get update -qq && apt-get install -y -qq curl unzip zip python3.12 python3.12-venv >/dev/null
if [ ! -x /opt/jdk/bin/java ]; then
  curl -sL -o /root/jdk.tgz "https://api.adoptium.net/v3/binary/version/jdk-25.0.3%2B9/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p /opt/jdk && tar -xzf /root/jdk.tgz -C /opt/jdk --strip-components=1
fi
export JAVA_HOME=/opt/jdk PATH=/opt/jdk/bin:/root/.jbang/bin:/root/.local/bin:$PATH
[ -x /root/.jbang/bin/jbang ] || curl -Ls https://sh.jbang.dev | bash -s - app setup >/dev/null
[ -x /root/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
cd $ROOT/comparator && uv sync -q
{
  echo "server: $(hostname), $(nproc) cpus"; lscpu | grep -E "Model name|Flags" | cut -c1-300; free -g | head -2
  java -version 2>&1; .venv/bin/python -c "import onnxruntime, torch; print('onnxruntime', onnxruntime.__version__, 'torch', torch.__version__)"
  echo "models: $MODELS_VERSION from Maven Central"
} > $EV/machine-box.txt 2>&1 || true
# Artifacts: pins.artifact downloads and digest-checks the graph and config.
.venv/bin/python -c "import pins; print(pins.artifact('onnx/paradee_int8.onnx')); print(pins.artifact('config.json'))"
ART=$ROOT/comparator/.artifacts
log "comparator, 1 thread, three passes"
THREADS="1" ./run_comparator_passes.sh $EV/comparator
log "comparator, 6 threads, three passes"
THREADS="6" ./run_comparator_passes.sh $EV/comparator
log "waiting for com.integrallis:models-audio:$MODELS_VERSION on Maven Central"
for i in $(seq 1 120); do
  curl -sf -o /dev/null "https://repo1.maven.org/maven2/com/integrallis/models-audio/$MODELS_VERSION/models-audio-$MODELS_VERSION.pom" && break
  sleep 60
done
curl -sf -o /dev/null "https://repo1.maven.org/maven2/com/integrallis/models-audio/$MODELS_VERSION/models-audio-$MODELS_VERSION.pom" || { log "release not on Central after two hours"; exit 1; }
log "candidate against the released library, three passes"
for pass in 1 2 3; do for corpus in wikitext103-heldout-200 harvard-200; do for voice in af_heart bf_isabella am_puck bm_fable; do
  out=$EV/candidate/$corpus/$voice/t1/pass$pass
  [ -f $out/pass.json ] && continue
  mkdir -p $out; uptime | sed 's/.*load averages*: *//' > $out/load.txt
  jbang -Dmodels.version=$MODELS_VERSION $ROOT/latency/ParadeeLatencyArm.java $ART/onnx_paradee_int8.onnx $ART/config.json $ROOT/corpus/$corpus.json $voice $pass $out 2>&1 | grep PARADEE_ARM
done; done; done
log "done; pass.json files under $EV"
cd $ROOT && tar czf /root/paradee-box-evidence.tgz --exclude='*.wav' --exclude='*.txt' evidence
