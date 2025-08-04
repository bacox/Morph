#!/bin/bash

prefix="nca"
echo "Running NCA all script"

# List of config files to run
bash_files=(
    DissDL/run_nca_all.sh
    EpidemicLearning/run_el-local.sh
    StaticTopologies/run.sh
)

# Loop through each bash file and execute it
for bash_file in "${bash_files[@]}"; do
    echo "Running $bash_file"
    # Check if the file exists
    if [[ -f "$bash_file" ]]; then
        cd "$(dirname "$bash_file")" || exit 1
        # echo "Changed directory to $(pwd)"
        # Execute the bash file
        # REmove parent from bash file path
        bash_file=$(basename "$bash_file")
        # echo "Executing $bash_file"
        
        bash "$bash_file" prefix="$prefix"

        # Go back to the original directory
        cd - || exit 1
        # echo "Returned to $(pwd)"
    else
        echo "Error: $bash_file not found."
        exit 1
    fi
done