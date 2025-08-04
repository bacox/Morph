#!/bin/bash

# Usage: ./run_backround.sh path/to/config.ini

# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <path-to-config-file>"
#     exit 1
# fi

# config_file="$1"

# if [ ! -f "$config_file" ]; then
#     echo "Error: Config file '$config_file' does not exist."
#     exit 1
# fi

# echo "Launching run.sh with config file: $config_file"
nohup bash run_el-oracle.sh &> nohup_exec.txt &
echo "Process started in background. Output redirected to nohup_exec.txt"
