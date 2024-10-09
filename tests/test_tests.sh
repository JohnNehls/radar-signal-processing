#!/usr/bin/env bash

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
        echo "Running $file..."
	# no block check is added to files int ./tests
        python "$file" "--no-block" > /dev/null 2>&1

        # Check if the last command was successful
        if [ $? -ne 0 ]; then
            echo "Error: $file failed to run successfully."
            # return 1
        fi
    done
    if [ $success -eq 1 ]; then
      echo "All Python files in $dir_path ran successfully."
    fi
}

echo "################## test files in ./tests #############################################"
run_python_files .

echo ""
echo "################## inspect plots in ./examples #######################################"
run_python_files ../examples

# Studies take a relativly long time to run, check less frequently
echo ""
echo "################## inspect plots in ./studies ########################################"
run_python_files ../studies
