#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
bash setup/run.sh scripts/human_edits/check_trial_v1_2.py c12b380a d17b4c89 2>&1 | grep -v "Warning\|warn(" | tail -3
