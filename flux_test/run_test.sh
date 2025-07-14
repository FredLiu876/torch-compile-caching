#!/bin/bash

# This script is used to run the test for the torch compile caching feature.
export TMPDIR=./local_cache
python full_test.py
rm -rf $TMPDIR
python run.py --phase run_compile
rm -rf $TMPDIR
python run.py --phase run_cache
python run.py --phase run