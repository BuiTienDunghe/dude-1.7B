#!/usr/bin/env bash
# Chấm decider-2b và adapter đối chứng trên 300 dòng chạy thử (batch 1). Không train.
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/bench_candidates/pilot; E=$R/eval; mkdir -p $E
CTRL=results/ft_loss/train/Qwen3-1.7B-s1/adapter
timeout 5400 bash setup/run.sh -X faulthandler scripts/bench/eval_decider_v1.py $R/pilot_rows.jsonl $E/decider > $E/log_decider.txt 2>&1
echo "decider exit=$?"
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $E --name control --adapter $CTRL --rows $R/pilot_rows.jsonl > $E/log_control.txt 2>&1
echo "control exit=$?"
ls $E/*/ $E/decider/*/ 2>/dev/null
