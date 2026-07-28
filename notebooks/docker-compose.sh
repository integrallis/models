#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker_config="$script_dir/.docker-public"
compose_file="$script_dir/docker-compose.yml"
host_docker_config="${DOCKER_CONFIG:-$HOME/.docker}"

if [[ -n "${DOCKER_HOST:-}" ]]; then
    docker_host="$DOCKER_HOST"
else
    docker_context="$(docker context show)"
    docker_host="$(docker context inspect "$docker_context" --format '{{.Endpoints.docker.Host}}')"
fi

plugin_dir="$docker_config/cli-plugins"
mkdir -p "$plugin_dir"

link_plugin() {
    local name="$1"
    local candidate
    for candidate in \
        "$host_docker_config/cli-plugins/$name" \
        "/usr/local/lib/docker/cli-plugins/$name" \
        "/usr/libexec/docker/cli-plugins/$name" \
        "/opt/homebrew/lib/docker/cli-plugins/$name" \
        "/Applications/Docker.app/Contents/Resources/cli-plugins/$name"; do
        if [[ -x "$candidate" ]]; then
            ln -sfn "$candidate" "$plugin_dir/$name"
            return
        fi
    done
}

link_plugin docker-buildx
link_plugin docker-compose

export DOCKER_CONFIG="$docker_config"
export DOCKER_HOST="$docker_host"
unset DOCKER_CONTEXT

if command -v docker-compose >/dev/null 2>&1; then
    compose=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
    compose=(docker compose)
else
    printf 'Docker Compose is required.\n' >&2
    exit 1
fi

exec "${compose[@]}" \
    --project-directory "$script_dir" \
    -f "$compose_file" \
    "$@"
