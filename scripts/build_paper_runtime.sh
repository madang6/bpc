#!/bin/sh
set -eu

archive=$(realpath "${1:?usage: $0 DARP_NATIVE_TAR [IMAGE_TAG]}")
tag=${2:-dpc-nldl-paper-buildout:latest}
dockerfile=$(realpath "$(dirname "$0")/../docker/paper-buildout.Dockerfile")
context=$(mktemp -d "$(dirname "$archive")/.dpc-nldl-paper-runtime.XXXXXX")
cleanup() {
    rm -f "$context/darp_native.tar.gz" "$context/Dockerfile"
    rmdir "$context"
}
trap cleanup EXIT
ln "$archive" "$context/darp_native.tar.gz"
cp "$dockerfile" "$context/Dockerfile"
docker build -t "$tag" "$context"
