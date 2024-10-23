#!/usr/bin/env bash

RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # no color

# Function to run Python files in a specified directory
run_python_files() {
    local dir_path=$1

    # Check if the directory exists
    if [ ! -d "$dir_path" ]; then
        echo "Error: Directory $dir_path does not exist."
        return 1
    fi

    success=1
    # Loop through all Python files in the specified directory
    for file in "$dir_path"/*.py
    do
        echo -e "${RED}Running $file...${NC}"
	# no block check is added to files int ./tests
	# Ignore stdout and warnings, print errors
        python -W ignore "$file" "--no-block" > /dev/null

        # Check if the last command was successful
        if [ $? -ne 0 ]; then
            echo "Error: $file failed to run successfully."
            success=0
        fi
    done
    if [ $success -eq 1 ]; then
      echo "All Python files in $dir_path ran successfully."
    fi
}

echo -e "${BLUE}################## test files in ./tests #####################################${NC}"
run_python_files .

# echo ""
# echo -e "${BLUE}################## inspect plots in ./examples ###############################${NC}"
# run_python_files ../examples

# # Studies take a relativly long time to run, check less frequently
# echo ""
# echo -e "${BLUE}################## inspect plots in ./studies ################################${NC}"
# run_python_files ../studies
