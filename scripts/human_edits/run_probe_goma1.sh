#!/usr/bin/env bash
# Đã gộp vào run_probe_user_item.sh. Giữ để lệnh cũ còn chạy: run_probe_goma1.sh [N] (N = 1 mặc định, hoặc 2).
exec bash /mnt/c/Users/dungbt06/Fine-tune/scripts/human_edits/run_probe_user_item.sh "results/bench_v1_2_trial/user_revised_probe/goma${1:-1}_user.json"
