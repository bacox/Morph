#!/bin/bash

# --- Parse prefix argument if present ---
if [[ "$1" == prefix=* ]]; then
    prefix="${1#prefix=}"

    # Strip quotes if present
    prefix="${prefix%\"}"
    prefix="${prefix#\"}"
    shift
fi
bash run.sh prefix="$prefix" config_baseline.ini config_gds.ini config_gms.ini
