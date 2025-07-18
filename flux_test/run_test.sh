#!/bin/bash

# This script is used to run the test for the torch compile caching feature.
python full_test.py
rm -rf /tmp/torchinductor_root/
python run.py --phase run_compile
cp -r /tmp/torchinductor_root/ ./local
rm -rf /tmp/torchinductor_root/
python run.py --phase run_cache
rm -rf /tmp/torchinductor_root/
cp -r ./local/ /tmp/torchinductor_root/
python run.py --phase run_compile
python run.py --phase run