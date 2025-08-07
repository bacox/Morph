#!/bin/bash

# --- Parse prefix argument if present ---
if [[ "$1" == prefix=* ]]; then
    prefix="${1#prefix=}"

    # Strip quotes if present
    prefix="${prefix%\"}"
    prefix="${prefix#\"}"
    shift
fi

decpy_path=../../eval # Path to eval folder
# graph=regular_100_3.edges # Absolute path of the graph file generated using the generate_graph.py script
# graph=fully-connected_100.edges
# graph=regular_100_7.edges
# graph=regular_100_14.edges
# graph=regular_200_3.edges
graph=regular_50_3.edges
run_path=../../eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config_static.ini
cp $graph $config_file $run_path

env_python=python # Path to python executable of the environment | conda recommended
machines=1 # number of machines in the runtime
iterations=8000
test_after=20
eval_file=testing.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=50 # 16 processes on 1 machine
echo procs per machine is $procs_per_machine

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
    prefix="${prefix}/${dataset}/${graph_name}/${alg_name}/"
else
    prefix=""
fi
log_dir=$run_path/$prefix$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder

echo "Using Log Directory: $log_dir"
# log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir