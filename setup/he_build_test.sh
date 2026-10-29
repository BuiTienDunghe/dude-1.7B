#!/usr/bin/env bash
# Chạy thử script dựng với lớp ghi đè CÓ chỉnh sửa (môi trường thử ở scratchpad), ra thư mục tạm; không in nội dung test.
cd /mnt/c/Users/dungbt06/Fine-tune
SD=/mnt/c/Users/dungbt06/AppData/Local/Temp/claude/C--Users-dungbt06-Fine-tune/a9d9eafb-bfb4-4b75-94b2-701deafa02ce/scratchpad
timeout 1800 bash setup/run.sh scripts/human_edits/build_revised.py --edits_dir $SD/apptest/data/human_edits --config_dir $SD/apptest/config --out_dir $SD/build_test_out 2>&1 | tail -2 | cut -c1-700
