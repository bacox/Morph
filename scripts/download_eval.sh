#!/bin/bash

# Remote servers
HOSTS=("st1" "st3")
# Source directory on remote
REMOTE_SUBDIR="topology-decl/eval"
# Local base directory to store pulled evals
LOCAL_BASE_DIR="eval_remote"
# Create local base directory if it doesn't exist
mkdir -p "$LOCAL_BASE_DIR"

mkdir -p "$LOCAL_BASE_DIR"

for HOST in "${HOSTS[@]}"; do
    DEST_DIR="$LOCAL_BASE_DIR/$HOST"

    mkdir -p "$DEST_DIR"
    echo "Pulling eval directory from $HOST:$REMOTE_SUBDIR to $DEST_DIR"
    
    rsync -Lravz --progress "$HOST:$REMOTE_SUBDIR/" "$DEST_DIR/"
done