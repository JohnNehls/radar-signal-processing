#!/usr/bin/env bash


export MPLBACKEND=Agg  # makes plt.show() becomes a no-op when running a python script

B_RED='\033[1;31m'
B_GREEN='\033[1;32m'
B_BLUE='\033[1;34m'
B_WHITE='\033[1m'
NC='\033[0m'

overall_success=1

# Function to run Python files in a specified directory
run_python_files() {
    local dir_path=$1

    # Check if the directory exists
    if ! [ -d "$dir_path" ]; then
        echo -e "${B_RED}Error: Directory $dir_path does not exist${NC}\n"
        return 1
    fi

    success=1
    # Loop through all Python files in the specified directory
    for file in "$dir_path"/*.py
    do
        # Skip files ending with _no_test.py
        if [[ "$file" == *_no_test.py ]]; then
            echo -e "${B_WHITE}Skipping $file${NC}"
            continue
        fi

        echo -e "${B_WHITE}Running $file...${NC}"
        # Ignore stdout and warnings, print errors
        python -W ignore "$file" > /dev/null

        # Check if the last command was successful
        if [ $? -ne 0 ]; then
            success=0
            overall_success=0
        fi
    done

    #Print if the run of the files in the direcotry was a success or not
    if [ $success -eq 1 ]; then
	echo -e "${B_GREEN}All Python files in $dir_path ran successfully${NC}\n"
    else
      echo -e "${B_RED}Atleast one error arose while running the Python files in $dir_path ${NC}\n"
    fi
}

# Get the path of the directory holding this script
DIR_NAME=$(dirname "$0")

echo -e "${B_BLUE}################## test files in ./exercises #####################################${NC}"
run_python_files ${DIR_NAME}/exercises

echo -e "${B_BLUE}################## inspect plots in ./rdms ###############################${NC}"
run_python_files ${DIR_NAME}/rdms

# Studies take a relativly long time to run, check less frequently
echo -e "${B_BLUE}################## inspect plots in ./studies ################################${NC}"
run_python_files ${DIR_NAME}/studies

if [ $overall_success -ne 1 ]; then
    exit 1
fi
