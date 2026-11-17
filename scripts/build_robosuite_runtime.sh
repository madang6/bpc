#!/bin/sh
set -eu

source=${1:?usage: $0 SIMULATOR_SOURCE_DIR [IMAGE_TAG]}
tag=${2:-bpc-robosuite:latest}
context=$(mktemp -d "$source/.bpc-robosuite.XXXXXX")
trap 'rm -rf "$context"' EXIT
cp "$(dirname "$0")/../docker/robosuite.Dockerfile" "$context/Dockerfile"
for package in robosuite robomimic mimicgen robosuite-task-zoo; do
    mkdir "$context/$package"
    cp -al "$source/$package/." "$context/$package"
    rm -rf "$context/$package/.git"
done
docker build -t "$tag" "$context"
