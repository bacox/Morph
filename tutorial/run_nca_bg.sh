#!/bin/bash

script=run_nca_all.sh
echo "Launching $script in background"
nohup bash "$script" &> nohup_exec.txt &
echo "Process started in background. Output redirected to nohup_exec.txt"
