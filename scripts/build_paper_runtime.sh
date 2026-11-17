#!/bin/sh
set -eu

archive=$(realpath "${1:?usage: $0 ENVIRONMENT_TAR [IMAGE_TAG]}")
tag=${2:-bpc-paper-buildout:latest}
dockerfile=$(realpath "$(dirname "$0")/../docker/paper-buildout.Dockerfile")
context=$(mktemp -d "$(dirname "$archive")/.bpc-paper-runtime.XXXXXX")
cleanup() {
    rm -f "$context/bpc_environment.tar.gz" "$context/Dockerfile" "$context/archive-files"
    rmdir "$context"
}
trap cleanup EXIT
tar -tzf "$archive" > "$context/archive-files"
if ! grep -Eq '^(\./)?bin/conda-unpack$' "$context/archive-files"; then
    echo "Use a relocatable conda-pack archive containing bin/conda-unpack." >&2
    exit 1
fi
rm "$context/archive-files"
ln "$archive" "$context/bpc_environment.tar.gz"
cp "$dockerfile" "$context/Dockerfile"
docker build -t "$tag" "$context"
