#!/bin/zsh
set -e
SCRIPT_DIR="${0:A:h}"
export MPLCONFIGDIR="$SCRIPT_DIR/.matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"
/usr/bin/env python3 "$SCRIPT_DIR/orbits.py" --show
