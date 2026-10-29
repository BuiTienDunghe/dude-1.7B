#!/usr/bin/env bash
# Dựng thử bench 1.2 (khu tạm; không đóng băng; không ghi data/). 1) kiểm đồng nhất với lớp ghi đè rỗng; 2) dựng thử từ lớp ghi đè thật.
cd /mnt/c/Users/dungbt06/Fine-tune
EMPTY=/mnt/c/Users/dungbt06/AppData/Local/Temp/claude/C--Users-dungbt06-Fine-tune/a9d9eafb-bfb4-4b75-94b2-701deafa02ce/scratchpad/empty_edits
bash setup/run.sh scripts/human_edits/build_revised.py --check_identity --edits_dir $EMPTY 2>&1 | grep -v "Warning\|warn(" | tail -1
echo "identity_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/build_revised.py --trial --nfc --provenance --test_labels results/human_review/label_decisions_remote.jsonl --out_dir results/bench_v1_2_trial 2>&1 | grep -v "Warning\|warn(" | tail -2
sha256sum data/bench_v1_1/dev_v1_1.jsonl data/bench_v1_1/test_v1.1.jsonl | cut -c1-16
