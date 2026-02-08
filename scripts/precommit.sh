#!/usr/bin/env bash

echo "Formatting with cargo fmt."
cargo +nightly fmt

echo "Linting with clippy."
cargo clippy --all-targets
