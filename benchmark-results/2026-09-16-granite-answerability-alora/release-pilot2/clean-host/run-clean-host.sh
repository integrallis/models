#!/usr/bin/env bash
# Clean-host Java 25 run of the Integrallis Granite 4.1 3B answerability aLoRA against the Maven
# Central Models 0.3.42 artifacts, producing the record ModelJars' requireCleanHostRun gate reads.
#
# Target: a freshly provisioned Ubuntu 24.04 host (x86_64 or aarch64), run as root, with this
# directory's AnswerabilityCleanHost.java and write_clean_host_run.py next to this script.
#
#   sudo bash run-clean-host.sh [output-dir]      (default output-dir: ./evidence)
#
# Outputs (commit them under release-pilot2/clean-host/evidence/):
#   fresh-check.txt          "absent|present <path>" for ~/.m2 ~/.jbang ~/.gradle, taken before
#                            anything else runs; freshMachine=true only if every line is "absent"
#   host-baseline.txt        uname, os-release, lscpu, free, df, nproc, preinstalled java/jbang
#   toolchain.txt            pinned JDK/JBang URLs and verified sha256s
#   java-version.txt         `java -version`
#   jbang-resolve.log        `jbang --verbose build` (dependency resolution from Maven Central)
#   resolved-classpath.txt   one line per dependency jar on `jbang info classpath`:
#                            "<sha256>  <jar filename>", LC_ALL=C sorted, each line newline-terminated.
#                            The script's own compiled jar (JBang cache) is excluded: it embeds build
#                            timestamps, so it is not a property of the resolution. Its hash is in
#                            program-sha256.txt together with the source hashes.
#                            resolvedClasspathSha256 = sha256 of this file's bytes.
#                            Resolved with JBang 0.141.0 from an empty repository on a dev machine
#                            (2026-09-17) this was 880b4f45a473abc15e937bafd29fcf2c4be79733281c0d1a39b17545a535ab47
#                            over backend-java, models-runtime, models-api 0.3.42, vectors-core 0.1.22,
#                            jackson-core 2.21.4, slf4j-api 2.0.17; a different value is a finding.
#   program-sha256.txt       sha256 of the script source, the writer and the compiled script jar
#   clean-host-output.log    stdout+stderr of the program run (this is outputLog)
#   clean-host-run.json      the gate record; outputLog.uri carries the placeholder
#                            <EVIDENCE_REVISION>, to be replaced by the commit containing the log
#   wrapper.log              this script's own progress lines
#
# Exit status: 0 only if clean-host-run.json has pass=true.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mkdir -p "${1:-$HERE/evidence}" && cd "${1:-$HERE/evidence}" && pwd)"
WORK="${WORK:-/opt/modeljars-clean-host}"
STORE="$WORK/store"
MODELS_VERSION=0.3.42
LOG_REPO_PATH="benchmark-results/2026-09-16-granite-answerability-alora/release-pilot2/clean-host/evidence/clean-host-output.log"

JDK_RELEASE='jdk-25.0.4.1+1'
JBANG_VERSION=0.141.0
JBANG_URL="https://github.com/jbangdev/jbang/releases/download/v$JBANG_VERSION/jbang-$JBANG_VERSION.tar"
JBANG_SHA256='0de5bb29a23159a6afc6f40dd510fbad05685827288ba7727c49d500719d0f92'
case "$(uname -m)" in
  x86_64)
    JDK_URL='https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz'
    JDK_SHA256='dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e' ;;
  aarch64)
    JDK_URL='https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_aarch64_linux_hotspot_25.0.4.1_1.tar.gz'
    JDK_SHA256='69df11a02cfa3ef7d7ca645e03edce6778ec090e100f6ae2b42097865730ac52' ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 64 ;;
esac

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(utc)" "$*" | tee -a "$OUT/wrapper.log"; }

# 1. Freshness, before anything can create a cache.
: > "$OUT/fresh-check.txt"
for cache in "$HOME/.m2" "$HOME/.jbang" "$HOME/.gradle"; do
  if [ -e "$cache" ]; then echo "present $cache"; else echo "absent $cache"; fi >> "$OUT/fresh-check.txt"
done
STARTED_AT="$(utc)"
log "start; fresh-check: $(tr '\n' ';' < "$OUT/fresh-check.txt")"

[ "$(id -u)" = 0 ] || { log "must run as root"; exit 64; }
for f in AnswerabilityCleanHost.java write_clean_host_run.py; do
  [ -f "$HERE/$f" ] || { log "missing $HERE/$f"; exit 64; }
done

# 2. Host baseline.
{
  echo "## date";         utc
  echo "## uname";        uname -a
  echo "## os-release";   cat /etc/os-release
  echo "## nproc";        nproc
  echo "## lscpu";        lscpu
  echo "## free";         free -h
  echo "## df";           df -h /
  echo "## preinstalled"; echo "java: $(command -v java || echo none)"; echo "jbang: $(command -v jbang || echo none)"
} > "$OUT/host-baseline.txt" 2>&1

# 3. Toolchain: pinned, checksum-verified, fail closed.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > "$OUT/apt.log" 2>&1
apt-get install -y -qq curl ca-certificates python3 tar >> "$OUT/apt.log" 2>&1

fetch() { # url dest sha256
  local url="$1" dest="$2" sha="$3" actual
  curl -fsSL --retry 5 --retry-delay 10 -o "$dest.part" "$url"
  actual="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then
    rm -f "$dest.part"; log "HASH MISMATCH $url expected $sha got $actual"; exit 2
  fi
  mv "$dest.part" "$dest"
}

mkdir -p "$WORK/toolchain" /opt/java /opt/jbang
fetch "$JDK_URL" "$WORK/toolchain/jdk.tar.gz" "$JDK_SHA256"
fetch "$JBANG_URL" "$WORK/toolchain/jbang.tar" "$JBANG_SHA256"
rm -rf "/opt/java/$JDK_RELEASE" "/opt/jbang/jbang-$JBANG_VERSION"
tar -xzf "$WORK/toolchain/jdk.tar.gz" -C /opt/java
tar -xf "$WORK/toolchain/jbang.tar" -C /opt/jbang
printf 'jdk %s %s %s\njbang %s %s %s\n' "$JDK_RELEASE" "$JDK_URL" "$JDK_SHA256" "$JBANG_VERSION" "$JBANG_URL" "$JBANG_SHA256" > "$OUT/toolchain.txt"

export JAVA_HOME="/opt/java/$JDK_RELEASE"
export PATH="$JAVA_HOME/bin:/opt/jbang/jbang-$JBANG_VERSION/bin:$PATH"
export JBANG_NO_VERSION_CHECK=true
java -version > "$OUT/java-version.txt" 2>&1
log "java: $(head -1 "$OUT/java-version.txt"); jbang: $(jbang --version 2>/dev/null)"

# 4. Resolve from Maven Central and fingerprint the classpath.
cp "$HERE/AnswerabilityCleanHost.java" "$WORK/AnswerabilityCleanHost.java"
cd "$WORK"
jbang --verbose build AnswerabilityCleanHost.java > "$OUT/jbang-resolve.log" 2>&1
CLASSPATH_RAW="$(jbang info classpath AnswerabilityCleanHost.java 2>/dev/null | tail -1)"
SCRIPT_JAR=""
: > "$OUT/resolved-classpath.unsorted"
IFS=':' read -r -a ENTRIES <<< "$CLASSPATH_RAW"
for entry in "${ENTRIES[@]}"; do
  [ -n "$entry" ] || continue
  [ -f "$entry" ] || { log "classpath entry is not a file: $entry"; exit 2; }
  case "$entry" in
    */cache/jars/AnswerabilityCleanHost.java.*/AnswerabilityCleanHost.jar) SCRIPT_JAR="$entry"; continue ;;
  esac
  printf '%s  %s\n' "$(sha256sum "$entry" | cut -d' ' -f1)" "$(basename "$entry")" >> "$OUT/resolved-classpath.unsorted"
done
LC_ALL=C sort "$OUT/resolved-classpath.unsorted" > "$OUT/resolved-classpath.txt"
rm -f "$OUT/resolved-classpath.unsorted"
[ -s "$OUT/resolved-classpath.txt" ] || { log "empty resolved classpath"; exit 2; }
{
  printf '%s  AnswerabilityCleanHost.java\n' "$(sha256sum "$HERE/AnswerabilityCleanHost.java" | cut -d' ' -f1)"
  printf '%s  write_clean_host_run.py\n' "$(sha256sum "$HERE/write_clean_host_run.py" | cut -d' ' -f1)"
  printf '%s  run-clean-host.sh\n' "$(sha256sum "$HERE/run-clean-host.sh" | cut -d' ' -f1)"
  if [ -n "$SCRIPT_JAR" ]; then
    printf '%s  AnswerabilityCleanHost.jar (compiled, not reproducible)\n' "$(sha256sum "$SCRIPT_JAR" | cut -d' ' -f1)"
  fi
} > "$OUT/program-sha256.txt"
log "resolved classpath: $(wc -l < "$OUT/resolved-classpath.txt") jars, sha256 $(sha256sum "$OUT/resolved-classpath.txt" | cut -d' ' -f1)"

# 5. The run.
COMMAND=(jbang AnswerabilityCleanHost.java --store "$STORE")
log "run: ${COMMAND[*]}"
set +e
"${COMMAND[@]}" 2>&1 | tee "$OUT/clean-host-output.log"
EXIT_CODE="${PIPESTATUS[0]}"
set -e
# Timestamps have one-second resolution and the gate requires completedAt > startedAt.
[ "$(utc)" != "$STARTED_AT" ] || sleep 1
COMPLETED_AT="$(utc)"
log "program exit $EXIT_CODE"

# 6. The gate record.
COMMAND_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${COMMAND[@]}")"
set +e
python3 "$HERE/write_clean_host_run.py" \
  --started-at "$STARTED_AT" --completed-at "$COMPLETED_AT" --exit-code "$EXIT_CODE" \
  --models-version "$MODELS_VERSION" --command-json "$COMMAND_JSON" \
  --java-version-file "$OUT/java-version.txt" --fresh-check-file "$OUT/fresh-check.txt" \
  --classpath-file "$OUT/resolved-classpath.txt" --output-log "$OUT/clean-host-output.log" \
  --output-log-repo-path "$LOG_REPO_PATH" --out "$OUT/clean-host-run.json"
WRITER_EXIT=$?
set -e
log "clean-host-run.json written; pass=$([ "$WRITER_EXIT" = 0 ] && echo true || echo false)"
exit "$WRITER_EXIT"
