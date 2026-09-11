#!/bin/zsh

set -euo pipefail

if (( $# != 0 )); then
    print -u2 "Usage: ./run.sh"
    exit 1
fi

cd "$(dirname "$0")"
exec make run
