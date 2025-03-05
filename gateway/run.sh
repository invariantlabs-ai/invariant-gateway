#!/bin/bash

# Validate configuration
python validate_config.py

# Check exit code of validation script
if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Exiting."
    exit 1
fi

if [ "$DEV_MODE" = "true" ]; then
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
else
    uvicorn serve:app --host 0.0.0.0 --port 8000
fi