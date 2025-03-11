#!/bin/bash

# Validate configuration
python validate_config.py

# Check exit code of validation script
if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Exiting."
    exit 1
fi

# using 'exec' belows ensures that signals like SIGTERM are passed to the child process
# and not the shell script itself (important when running in a container)
if [ "$DEV_MODE" = "true" ]; then
    exec uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
else
    exec uvicorn serve:app --host 0.0.0.0 --port 8000
fi