#!/bin/bash

# if DEV_MODE is true, then run the app with auto-reload
if [ "$DEV_MODE" = "true" ]; then
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
else
    uvicorn serve:app --host 0.0.0.0 --port 8000
fi
