#!/usr/bin/env bash
# Bản XEM TRƯỚC bench 1.1 đã áp quyết định cuối (lớp ghi đè trên bản sao ở scratchpad), chỉ để Opus đọc lại. Không ghi vào data/.
cd /mnt/c/Users/dungbt06/Fine-tune
E=/mnt/c/Users/dungbt06/AppData/Local/Temp/claude/C--Users-dungbt06-Fine-tune/37a42dde-25b8-49c2-a6df-b7c150c1a0b0/scratchpad/final_preview/edits
timeout 1800 bash setup/run.sh scripts/human_edits/build_revised.py --edits_dir "$E" --out_dir results/final_check/preview > results/final_check/preview_build.log 2>&1
echo "exit=$?"; tail -3 results/final_check/preview_build.log | cut -c1-2000
