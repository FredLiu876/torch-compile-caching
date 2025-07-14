#!/bin/bash

# This script is used to run the test for the torch compile caching feature.
export TORCHINDUCTOR_CACHE_DIR=./local_cache
python full_test.py
rm -rf $TORCHINDUCTOR_CACHE_DIR
python run_compile.py --cache_path cache
rm -rf $TORCHINDUCTOR_CACHE_DIR
python run_cache.py --cache_path cache
python run.py