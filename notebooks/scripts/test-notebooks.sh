#!/usr/bin/env bash

set -euo pipefail

repository=${MODELS_REPOSITORY:-/home/jovyan/work/models}
notebook_dir="$repository/notebooks"
output_dir=${MODELS_NOTEBOOK_OUTPUT_DIR:-"$repository/build/notebooks/executed"}
mode=${MODELS_NOTEBOOK_MODE:-source}

case "$mode" in
    source|release)
        notebooks=(
            "$notebook_dir"/01_getting_started.ipynb
            "$notebook_dir"/02_framework_adapters.ipynb
            "$notebook_dir"/03_guarded_rag.ipynb
        )
        ;;
    *)
        printf 'Unsupported MODELS_NOTEBOOK_MODE: %s (expected source or release)\n' "$mode" >&2
        exit 2
        ;;
esac

mkdir -p "$output_dir"

for notebook in "${notebooks[@]}"; do
    name=$(basename "$notebook")
    printf 'Executing %s (%s dependencies)\n' "$name" "$mode"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.kernel_name=java \
        --ExecutePreprocessor.timeout=600 \
        --output-dir="$output_dir" \
        --output="$name" \
        "$notebook"
done

python "$notebook_dir/scripts/validate_notebook_outputs.py" "$output_dir"

printf 'Executed %d notebook(s); outputs: %s\n' "${#notebooks[@]}" "$output_dir"
