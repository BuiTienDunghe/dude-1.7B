#!/usr/bin/env bash
# Bước D (1/3): dựng bench 1.2 từ lớp ghi đè hiện tại vào KHU TẠM results/bench_v1_2_stage/ (chưa đóng băng, chưa ghi data/), rồi kiểm.
# 1) kiểm đồng nhất với lớp ghi đè rỗng; 2) dựng (--trial: ghi ngoài data/, không chmod); 3) check_stage_v1_2.py (không dấu, rò rỉ, provenance).
cd /mnt/c/Users/dungbt06/Fine-tune
EMPTY=/mnt/c/Users/dungbt06/AppData/Local/Temp/claude/C--Users-dungbt06-Fine-tune/a9d9eafb-bfb4-4b75-94b2-701deafa02ce/scratchpad/empty_edits
S=results/bench_v1_2_stage
rm -rf $S; mkdir -p $S
bash setup/run.sh scripts/human_edits/build_revised.py --check_identity --edits_dir $EMPTY 2>&1 | grep -v "Warning\|warn(" | tail -1 | cut -c1-300
echo "identity_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/build_revised.py --trial --nfc --provenance --test_labels results/human_review/label_decisions_remote.jsonl --out_dir $S 2>&1 | grep -v "Warning\|warn(" | tail -1
echo "build_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/check_stage_v1_2.py $S 2>&1 | grep -v "Warning\|warn(" | tail -2
echo "check_exit=${PIPESTATUS[0]}"
sha256sum data/bench_v1_1/dev_v1_1.jsonl data/bench_v1_1/test_v1.1.jsonl data/human_edits/edits_log.jsonl | cut -c1-16
