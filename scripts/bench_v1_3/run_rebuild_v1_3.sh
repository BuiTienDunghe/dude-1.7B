#!/usr/bin/env bash
# Bench 1.3, đóng băng (1/2): dựng LẠI từ lớp ghi đè vào results/bench_v1_3_trial/rebuild (như bản thử) và chạy kiểm. Bước 2: freeze_v1_3.py.
cd /mnt/c/Users/dungbt06/Fine-tune
S=results/bench_v1_3_trial/rebuild
rm -rf $S; mkdir -p $S
bash setup/run.sh scripts/human_edits/build_revised.py --trial --nfc --provenance --test_labels results/human_review/label_decisions_remote.jsonl --out_dir $S 2>&1 | grep -v "Warning\|warn(" | tail -1 | cut -c1-300
echo "build_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/check_stage_v1_2.py $S 2>&1 | grep -v "Warning\|warn(" | tail -1 | cut -c1-300
echo "check_exit=${PIPESTATUS[0]}"
