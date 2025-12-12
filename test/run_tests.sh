#!/bin/bash
# Can be run from either the repo root or the test directory
# Example usage: sh ./run_tests.sh [additional pytest arguments]

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine the test directory
if [[ "$(basename "$SCRIPT_DIR")" == "test" ]]; then
    # Script is in test directory
    TEST_DIR="$SCRIPT_DIR"
else
    # Script is in repo root
    TEST_DIR="$SCRIPT_DIR/test"
fi

cd "$TEST_DIR"
python -m pytest "$@"
