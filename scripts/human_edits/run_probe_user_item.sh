#!/usr/bin/env bash
# Chấm thử MỘT câu dev với bản dịch người dùng (chưa vào lớp ghi đè). Tham số: đường dẫn JSON (tương đối thư mục dự án), ví dụ
#   results/bench_v1_2_trial/user_revised_probe/grandcanyon1_user.json  -> ra user_revised_probe/grandcanyon1/
# Dựng dòng, chấm 7 model cùng điều kiện bản thử. CHỈ DEV; KHÔNG test, KHÔNG train, không ghi data/.
cd /mnt/c/Users/dungbt06/Fine-tune
J=$1; [ -s "$J" ] || { echo "thiếu file JSON: $J"; exit 1; }
NAME=$(basename "$J" .json); NAME=${NAME%_user}
G=results/bench_v1_2_trial/user_revised_probe/$NAME; RS=$G/${NAME}_dev.jsonl; E=$G/eval
rm -rf $E; mkdir -p $E
bash setup/run.sh scripts/human_edits/probe_user_item.py build "$J" 2>&1 | grep -v "Warning\|warn(" | tail -1
[ -s $RS ] || { echo "build failed"; exit 1; }
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; STRIP=results/ft_strip/train/Qwen3-1.7B-s1/adapter; FTD=results/ft_decider/train/decider-2b-s1/adapter
ev_dec() { timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $RS $E/$1 ${2:+--adapter $2} > $E/log_$1.txt 2>&1; echo "dec $1 exit=$?"; }
ev_base() { timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 ${2:+--adapter $2} --rows $RS > $E/log_$1.txt 2>&1; echo "base $1 exit=$?"; }
ev_dec decider
ev_dec ftd $FTD
ev_base control $CTRL
ev_base base
ev_base strip $STRIP
echo "PROBE_${NAME}_DONE"
