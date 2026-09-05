#!/usr/bin/env bash

if [[ "${MODELS_NOTEBOOK_PREPARE:-true}" != "true" ]]; then
    return 0
fi

repository=/home/jovyan/work/models
mode=${MODELS_NOTEBOOK_MODE:-source}
version=${MODELS_VERSION:-0.3.29}
repository_url=${MODELS_NOTEBOOK_REPOSITORY:-}

case "$mode" in
    source)
        if ! "$repository/gradlew" \
            --no-daemon \
            -p "$repository" \
            prepareNotebookClasspath \
            -PnotebookMode=source; then
            return 1
        fi
        ;;
    release)
        args=(
            --no-daemon
            -p "$repository"
            prepareNotebookClasspath
            -PnotebookMode=release
            "-PnotebookVersion=$version"
        )
        if [[ -n "$repository_url" ]]; then
            args+=("-PnotebookRepository=$repository_url")
        fi
        if ! "$repository/gradlew" "${args[@]}"; then
            return 1
        fi
        ;;
    *)
        printf 'Unsupported MODELS_NOTEBOOK_MODE: %s (expected source or release)\n' "$mode" >&2
        return 2
        ;;
esac

unset repository mode version repository_url args
