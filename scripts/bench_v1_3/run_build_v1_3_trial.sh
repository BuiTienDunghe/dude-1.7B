#!/usr/bin/env bash
# Bench 1.3, DỰNG THỬ (khu tạm results/bench_v1_3_trial/; không ghi data/, không đóng băng): dựng từ lớp ghi đè hiện tại như 1.2
# (build_revised --trial --nfc --provenance --test_labels ...; có chốt kiểm dev↔test), rồi check_stage_v1_2.py (không dấu, rò rỉ, provenance).
# Phần mơ hồ và so với 1.2: scripts/bench_v1_3/make_v1_3_trial.py (Python Windows).
cd /mnt/c/Users/dungbt06/Fine-tune
S=results/bench_v1_3_trial/base
rm -rf $S; mkdir -p $S
bash setup/run.sh scripts/human_edits/build_revised.py --trial --nfc --provenance --test_labels results/human_review/label_decisions_remote.jsonl --out_dir $S 2>&1 | grep -v "Warning\|warn(" | tail -1 | cut -c1-400
echo "build_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/check_stage_v1_2.py $S 2>&1 | grep -v "Warning\|warn(" | tail -2 | cut -c1-600
echo "check_exit=${PIPESTATUS[0]}"
