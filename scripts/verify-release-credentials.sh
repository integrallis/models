#!/usr/bin/env bash
set -euo pipefail

signing_only=false
if [[ "${1:-}" == "--signing-only" ]]; then
  signing_only=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--signing-only]" >&2
  exit 2
fi

required=(
  GPG_PUBLIC_KEY
  GPG_SECRET_KEY
  GPG_PASSPHRASE
)
if [[ "$signing_only" == false ]]; then
  required+=(
    MAVENCENTRAL_USERNAME
    MAVENCENTRAL_PASSWORD
  )
fi

for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "::error::Required release secret $name is empty or unavailable." >&2
    exit 1
  fi
done

for name in GPG_PASSPHRASE MAVENCENTRAL_USERNAME MAVENCENTRAL_PASSWORD; do
  if [[ -n "${!name:-}" && ( "${!name}" == *$'\r'* || "${!name}" == *$'\n'* ) ]]; then
    echo "::error::Release secret $name contains a CR or LF character." >&2
    exit 1
  fi
done

temp_root="${RUNNER_TEMP:-/tmp}"
work_dir="$(mktemp -d "$temp_root/models-release-credentials.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

public_home="$work_dir/public"
secret_home="$work_dir/secret"
install -d -m 0700 "$public_home" "$secret_home"

if ! printf '%s\n' "$GPG_PUBLIC_KEY" |
  gpg --homedir "$public_home" --batch --quiet --import >/dev/null 2>&1; then
  echo "::error::GPG_PUBLIC_KEY is not an importable armored public key." >&2
  exit 1
fi
if ! printf '%s\n' "$GPG_SECRET_KEY" |
  gpg --homedir "$secret_home" --batch --quiet --import >/dev/null 2>&1; then
  echo "::error::GPG_SECRET_KEY is not an importable armored secret key." >&2
  exit 1
fi

public_fingerprint="$(
  gpg --homedir "$public_home" --batch --with-colons --fingerprint |
    awk -F: '$1 == "fpr" { print $10; exit }'
)"
secret_fingerprint="$(
  gpg --homedir "$secret_home" --batch --with-colons --list-secret-keys --fingerprint |
    awk -F: '$1 == "fpr" { print $10; exit }'
)"
if [[ -z "$public_fingerprint" || -z "$secret_fingerprint" ]]; then
  echo "::error::A release-key fingerprint could not be determined." >&2
  exit 1
fi
if [[ "$public_fingerprint" != "$secret_fingerprint" ]]; then
  echo "::error::GPG_PUBLIC_KEY and GPG_SECRET_KEY are not the same key pair." >&2
  exit 1
fi

probe="$work_dir/probe.txt"
signature="$work_dir/probe.txt.asc"
printf 'Models release credential preflight\n' >"$probe"
if ! printf '%s' "$GPG_PASSPHRASE" |
  gpg \
    --homedir "$secret_home" \
    --batch \
    --yes \
    --pinentry-mode loopback \
    --passphrase-fd 0 \
    --local-user "$secret_fingerprint" \
    --armor \
    --detach-sign \
    --output "$signature" \
    "$probe" >/dev/null 2>&1; then
  echo "::error::GPG_PASSPHRASE cannot unlock GPG_SECRET_KEY for signing." >&2
  exit 1
fi
if ! gpg \
  --homedir "$public_home" \
  --batch \
  --verify "$signature" "$probe" >/dev/null 2>&1; then
  echo "::error::The release signing probe could not be verified." >&2
  exit 1
fi

if [[ "$signing_only" == false ]]; then
  authorization="$(
    printf '%s:%s' "$MAVENCENTRAL_USERNAME" "$MAVENCENTRAL_PASSWORD" |
      base64 |
      tr -d '\r\n'
  )"
  central_url="${CENTRAL_PUBLISHER_URL:-https://central.sonatype.com/api/v1/publisher}"
  http_status="$(
    curl \
      --silent \
      --show-error \
      --output "$work_dir/central-response.txt" \
      --write-out '%{http_code}' \
      --request POST \
      --header "Authorization: Bearer $authorization" \
      "$central_url/status?id=00000000-0000-0000-0000-000000000000"
  )"
  case "$http_status" in
    401 | 403)
      echo "::error::Maven Central rejected the configured user token (HTTP $http_status)." >&2
      exit 1
      ;;
    000 | 5??)
      echo "::error::Maven Central credential probe failed with HTTP $http_status." >&2
      exit 1
      ;;
  esac
fi

echo "Release credential preflight passed."
