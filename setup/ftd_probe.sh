#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
timeout 600 bash setup/run.sh scripts/ft_decider/probe_modules.py 2>&1 | tail -3 | cut -c1-1500
