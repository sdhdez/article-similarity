#!/usr/bin/env bash
pip install --user -U -e .
/run_jupyterlab.sh --allow-root --no-browser --ip=0.0.0.0
