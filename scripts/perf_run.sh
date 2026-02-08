#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export RUST_LOG=trace

# To record data for later analysis.
#sudo perf record -e cache-references,cache-misses -g -- ./target/release/image_processing "${@:1}"
# Then:
#  perf report -v

# To just display a list of statistics.
sudo perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations "$PROJECT_ROOT"/target/release/image_processing "${@:1}"

# e.g., to test:
# ./run_perf --path /home/sean/Code_projects/wgpu_grapher/scratch/images/pexels-shottrotter-32654688.jpg --output-path scratch/sobel.jpg sobel
