#!/usr/bin/env bash
# Remote-only, sequential matched runs. No admission verdict from smoke/encoder evidence alone.
set -euo pipefail
[[ $(uname -s) == Linux ]] || { echo 'Run real-model qualification on remote Linux' >&2; exit 2; }
[[ $# == 3 ]] || { echo 'Usage: run.sh models-bench-distribution candidate-commit output-directory' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
distribution=$(realpath "$1")
revision=$2
[[ $revision =~ ^[0-9a-f]{40}$ ]] || { echo 'Supply a full immutable candidate commit' >&2; exit 2; }
mkdir -p "$3"
output=$(realpath "$3")
[[ ! -e $output/results ]] || { echo 'Refusing to reuse an existing result directory' >&2; exit 2; }
python=${VISION_PYTHON:-python3}
java=${VISION_JAVA:-java}
threads=${VISION_THREADS:-8}
artifacts=${VISION_ARTIFACT_DIRECTORY:-$output/artifacts}
mkdir -p "$output/results"
"$python" -m unittest discover -s "$script_dir" -p 'test_*.py'
"$python" "$script_dir/fetch.py" "$artifacts"
"$python" "$script_dir/images.py" "$output/images"
"$python" -m pip freeze > "$output/python-environment.txt"
"$java" -version > "$output/java-version.txt" 2>&1
lscpu > "$output/cpu.txt"
sha256sum "$distribution"/lib/*.jar > "$output/payload-sha256.txt"
printf '%s\n' "$revision" > "$output/candidate-commit.txt"
mmproj=$artifacts/llava-v1.5-7b-mmproj-model-f16.gguf
text=$artifacts/llava-v1.5-7b-Q4_K_M.gguf
for probe in cats wide-gradient tall-bands; do
    image=$output/images/$probe.png
    "$python" "$script_dir/oracle.py" "$mmproj" "$image" "$output/results/$probe/oracle" --threads "$threads" --warmups 1 --repetitions 3
    extra=()
    [[ $probe == cats ]] && extra=("$text")
    "$java" --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx12g \
        -Dvectors.gguf.threads="$threads" -Dmodels.purejava.maxContextLength=1024 \
        -Dmodels.qualification.revision="$revision" -cp "$distribution/lib/*" \
        com.integrallis.models.bench.VisionQualificationCli \
        "$mmproj" "$image" "$output/results/$probe/candidate" 1 3 "${extra[@]}"
    "$python" "$script_dir/compare.py" "$output/results/$probe/candidate" "$output/results/$probe/oracle" "$output/results/$probe/verdict.json"
done
printf 'INTEGRATION_PASS\n' > "$output/STATUS"
