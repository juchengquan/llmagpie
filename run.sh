#!/bin/bash
set -e

source activate v10

uvicorn test_api:app \
    --port 8080
    # --reload