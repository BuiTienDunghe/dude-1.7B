#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 900 bash setup/run.sh scripts/restore/tok_ratio.py 2>&1 | tail -3 | cut -c1-250
