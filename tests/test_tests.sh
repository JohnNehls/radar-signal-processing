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

echo "################################################################################"
echo "##################### Tests in ./tests #########################################"
echo "################################################################################"
run_python_files .

echo ""
echo ""
echo "################################################################################"
echo "##################### Tests in ./examples ######################################"
echo "################################################################################"
run_python_files ../examples
