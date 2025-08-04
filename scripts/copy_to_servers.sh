#!/bin/bash

# Remote path where the code should be synced
REMOTE_PATH="topology-decl"
# Remote hosts
HOSTS=("st1" "st3")

# Build rsync include/exclude rules from .gitignore
RSYNC_EXCLUDES=$(git ls-files --others --ignored --exclude-standard --directory)
EXCLUDE_ARGS=()

for path in $RSYNC_EXCLUDES; do
    EXCLUDE_ARGS+=("--exclude=$path")
done

# Always exclude the eval directory
EXCLUDE_ARGS+=("--exclude=eval" "--exclude=eval_remote")

# Sync to each host
for HOST in "${HOSTS[@]}"; do
    echo "Syncing to $HOST:$REMOTE_PATH"
    rsync -ravz --progress "${EXCLUDE_ARGS[@]}" ./ "$HOST:$REMOTE_PATH"
done