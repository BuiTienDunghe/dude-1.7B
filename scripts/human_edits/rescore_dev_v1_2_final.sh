#!/usr/bin/env bash
# Bước D: chấm lại DEV 1.2 (bản dựng cuối ở khu tạm) — chỉ các dòng có nội dung khác bản thử cuối (results/bench_v1_2_stage/rescore/rescore_dev.jsonl,
# do diff_stage_vs_trial_v1_2.py ghi). Cùng điều kiện như lần chấm 1.1 / bản thử. KHÔNG chạy test, KHÔNG train.
cd /mnt/c/Users/dungbt06/Fine-tune
S=results/bench_v1_2_stage; RS=$S/rescore/rescore_dev.jsonl; E=$S/eval_rescore
[ -s $RS ] || { echo "thiếu $RS"; exit 1; }
rm -rf $E; mkdir -p $E
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; STRIP=results/ft_strip/train/Qwen3-1.7B-s1/adapter; FTD=results/ft_decider/train/decider-2b-s1/adapter
ev_dec() { timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $RS $E/$1 ${2:+--adapter $2} > $E/log_$1.txt 2>&1; echo "dec $1 exit=$?"; }
ev_base() { timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 ${2:+--adapter $2} --rows $RS > $E/log_$1.txt 2>&1; echo "base $1 exit=$?"; }
ev_dec decider
ev_dec ftd $FTD
ev_base control $CTRL
ev_base base
ev_base strip $STRIP
echo RESCORE_DONE
