#!/bin/sh
set -eu

source=${1:?usage: $0 OPENPI_SOURCE [TAG]}
tag=${2:-bpc-robosuite-pi05:latest}
revision=15a9616a00943ada6c20a0f158e3adb39df2ccac
test "$(git -C "$source" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$source" status --porcelain)"
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$(dirname "$0")/../docker/robosuite_pi05.Dockerfile" "$context/Dockerfile"
cp -a "$source" "$context/openpi"
rm -rf "$context/openpi/.git" "$context/openpi/.venv"
docker build -t "$tag" "$context"
