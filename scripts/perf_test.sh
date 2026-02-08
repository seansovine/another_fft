#!/usr/bin/env bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export RUST_LOG=trace

function run_test {
	"$PROJECT_ROOT"/target/release/"$1"
	sudo perf stat -dB "$PROJECT_ROOT"/target/release/"$1"
}

# function perf_permissions {
#     sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
#     sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
#     sudo sh -c 'echo 0 > /proc/sys/kernel/nmi_watchdog'
# }

RUSTFLAGS='-C force-frame-pointers=y' cargo build --release

run_test perf_test_1
echo -e "---------------------------------\n"
run_test perf_test_2
