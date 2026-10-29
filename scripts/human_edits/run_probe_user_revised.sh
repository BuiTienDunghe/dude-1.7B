#!/usr/bin/env bash
# Thử lại 6 cặp dev người dùng tự dịch (Buffalo, Harper, Canyoning#2, sib200:810): dựng tạm từ lớp ghi đè HIỆN TẠI (khu tạm, không ghi data/), lấy 18 dòng dev, chấm 7 model
# cùng điều kiện như bản dựng thử (setup/b11_eval.sh). CHỈ DEV; KHÔNG chạy test, KHÔNG train.
cd /mnt/c/Users/dungbt06/Fine-tune
SPW=/mnt/c/Users/dungbt06/AppData/Local/Temp/claude/C--Users-dungbt06-Fine-tune/a9d9eafb-bfb4-4b75-94b2-701deafa02ce/scratchpad/probe_build
P=results/bench_v1_2_trial/user_revised_probe; RS=$P/probe_dev.jsonl; E=$P/eval
rm -rf $SPW $E; mkdir -p $P $E
bash setup/run.sh scripts/human_edits/build_revised.py --trial --nfc --provenance --test_labels results/human_review/label_decisions_remote.jsonl --out_dir $SPW 2>&1 | grep -v "Warning\|warn(" | tail -1 | cut -c1-200
echo "build_exit=${PIPESTATUS[0]}"
bash setup/run.sh scripts/human_edits/probe_user_revised.py extract $SPW || exit 1
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter; STRIP=results/ft_strip/train/Qwen3-1.7B-s1/adapter; FTD=results/ft_decider/train/decider-2b-s1/adapter
ev_dec() { timeout 3600 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $RS $E/$1 ${2:+--adapter $2} > $E/log_$1.txt 2>&1; echo "dec $1 exit=$?"; }
ev_base() { timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name $1 ${2:+--adapter $2} --rows $RS > $E/log_$1.txt 2>&1; echo "base $1 exit=$?"; }
ev_dec decider
ev_dec ftd $FTD
ev_base control $CTRL
ev_base base
ev_base strip $STRIP
echo PROBE_DONE
