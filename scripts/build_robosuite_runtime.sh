#!/bin/sh
set -eu

source=${1:-/media/admin/data/StanfordMSL/sdp-eval/deps}
tag=${2:-dpc-nldl-robosuite:latest}
context=$(mktemp -d "$source/.dpc-nldl-robosuite.XXXXXX")
trap 'rm -rf "$context"' EXIT
cp "$(dirname "$0")/../docker/robosuite.Dockerfile" "$context/Dockerfile"
for package in robosuite robomimic mimicgen robosuite-task-zoo; do
    mkdir "$context/$package"
    cp -al "$source/$package/." "$context/$package"
    rm -rf "$context/$package/.git"
done
docker build -t "$tag" "$context"
