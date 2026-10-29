#!/usr/bin/env bash
# Chấm một file dòng DEV bằng 7 model (cùng điều kiện như lần chấm 1.1 / bản thử / bản đóng băng). Dùng: rescore_rows_v1_2.sh <file dòng> <thư mục ra>.
# CHỈ DEV; KHÔNG chạy test, KHÔNG train.
cd /mnt/c/Users/dungbt06/Fine-tune
RS=$1; E=$2
[ -s "$RS" ] || { echo "thiếu $RS"; exit 1; }
case "$RS" in *test*) echo "từ chối: tên file có 'test'"; exit 1;; esac
rm -rf $E; mkdir -p $E
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; STRIP=results/ft_strip/train/Qwen3-1.7B-s1/adapter; FTD=results/ft_decider/train/decider-2b-s1/adapter
ev_dec() { timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $RS $E/$1 ${2:+--adapter $2} > $E/log_$1.txt 2>&1; echo "dec $1 exit=$?"; }
ev_base() { timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 ${2:+--adapter $2} --rows $RS > $E/log_$1.txt 2>&1; echo "base $1 exit=$?"; }
ev_dec decider
ev_dec ftd $FTD
ev_base control $CTRL
ev_base base
ev_base strip $STRIP
echo RESCORE_ROWS_DONE
