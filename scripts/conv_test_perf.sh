#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export RUST_LOG=trace

function run_test {
	$PROJECT_ROOT/target/release/image_processing \
		--path /home/sean/Code_projects/wgpu_grapher/scratch/images/pexels-shottrotter-32654688.jpg \
		--output-path "$PROJECT_ROOT"/scratch/sobel.jpg \
		conv-test $1

	sudo perf stat -dB "$PROJECT_ROOT"/target/release/image_processing \
		--path /home/sean/Code_projects/wgpu_grapher/scratch/images/pexels-shottrotter-32654688.jpg \
		--output-path "$PROJECT_ROOT"/scratch/sobel.jpg \
		conv-test $1
}

# function perf_permissions {
#     sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
#     sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
#     sudo sh -c 'echo 0 > /proc/sys/kernel/nmi_watchdog'
# }

RUSTFLAGS='-C force-frame-pointers=y' cargo build --release
echo ""

run_test test1
echo -e "---------------------------------\n"
run_test naive1
