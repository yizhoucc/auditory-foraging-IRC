#!/bin/bash

# Define your Conda environment name
CONDA_ENV="IRC"

# List of commands to run in each screen
k=10
COMMANDS=()
for ((i = 1; i <= k; i++)); do
    COMMANDS+=("python train_$i.py")
done
# COMMANDS=("python script1.py" "python script2.py" "python script3.py")

# Start a new screen session with a custom session name
screen -d -m -S no_attention_control

# Function to create a new screen and run a command
create_screen() {
    local cmd="$1"
    screen -S no_attention_control -X screen -t "$cmd" bash -c "conda activate $CONDA_ENV && $cmd"
}

# Activate Conda environment and run the first command in the first screen
create_screen "${COMMANDS[0]}"

# Create and run commands in additional screens
for ((i = 1; i < ${#COMMANDS[@]}; i++)); do
    create_screen "${COMMANDS[$i]}"
done

# Attach to the screen session to view the screens
screen -r no_attention_control