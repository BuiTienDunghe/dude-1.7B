#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 900 nice -n 19 bash setup/run.sh scripts/survey/sample_lengths.py 2>&1 | tail -25 | cut -c1-200
