#!/bin/bash

# This script is used to run the test for the torch compile caching feature.
python full_test.py
#rm -rf /tmp/torchinductor_root/
python run.py --phase run_compile
#rm -rf /tmp/torchinductor_root/
python run.py --phase run_cache
python run.py --phase run