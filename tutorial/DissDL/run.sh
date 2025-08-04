#!/bin/bash
set -e
set -o pipefail

# --- Parse prefix argument if present ---
if [[ "$1" == prefix=* ]]; then
    prefix="${1#prefix=}"

    # Strip quotes if present
    prefix="${prefix%\"}"
    prefix="${prefix#\"}"
    shift
fi

# --- Validate input arguments ---
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config1.ini> [config2.ini ...]"
    exit 1
fi

decpy_path=../../eval
# graph=regular_100_3.edges
graph=fully-connected_100.edges
procs_per_machine=100
run_path=../../eval/data
env_python=python
machines=1
iterations=8000
test_after=20
eval_file=testingDissDL.py
log_level=INFO
m=0



# Needed parameters
# -graph_name
# -dataset
# -algorithm

# --- Verify graph file exists ---
if [ ! -f "$graph" ]; then
    echo "Error: Graph file '$graph' not found."
    exit 1
fi

for config_file in "$@"; do
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file '$config_file' not found. Skipping..."
        
        # Raise error

        echo "Exiting due to missing config file."
        exit 1
    fi
done


for config_file in "$@"; do
    config_base=$(basename "$config_file" .ini)
    timestamp=$(date '+%Y-%m-%dT%H:%M')

    # Get stem of graph
    if [[ "$graph" == *.* ]]; then
        graph_name=$(basename "$graph" ".edges")
    else
        graph_name=$graph
    fi

    # Remove .ini extension from config file and split on underscore, take last part
    alg_name=$(basename "$config_file" .ini | awk -F'_' '{print $NF}')

    # Extract dataset from config file
    dataset=$(awk -F "=" '/dataset_class/ {print $2}' "$config_file" | tr -d '[:space:]')

    # Construct prefix if provided
    if [[ -n "$prefix" ]]; then
        prefix_text="${prefix}/${dataset}/${graph_name}/${alg_name}/"
    else
        prefix_text=""
    fi
    log_dir=$run_path/$prefix_text$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
    
    echo "Running with log directory: $log_dir"

    echo "Running with config: $config_file"
    echo "Machine ID: $m | Procs per machine: $procs_per_machine"
    echo "Log directory: $log_dir"

    mkdir -p "$log_dir"

    cp "$graph" "$config_file" "$run_path"

    echo "Started running at $(date)"
    START_T=$(date +%s.%N)

    # Echo the command being run
    echo "Executing: $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf ${run_path}/$(basename "$config_file") -ll $log_level -wsd $log_dir"

    $env_python $eval_file \
        -ro 0 \
        -tea $test_after \
        -ld "$log_dir" \
        -mid $m \
        -ps $procs_per_machine \
        -ms $machines \
        -is $iterations \
        -gf "$run_path/$graph" \
        -ta $test_after \
        -cf "$run_path/$(basename "$config_file")" \
        -ll $log_level \
        -wsd "$log_dir"

    END_T=$(date +%s.%N)
    DIFF_T=$(echo "$END_T - $START_T" | bc)
    echo "Finished at $(date)"
    echo "Execution took $DIFF_T seconds"
    echo "-----------------------------------------------"
done