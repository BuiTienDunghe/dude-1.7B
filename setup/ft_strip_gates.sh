#!/usr/bin/env bash
# Cổng G1–G3 trước khi train thử có bỏ dấu ngẫu nhiên. Mọi bước có timeout. Không train.
set -uo pipefail
cd /mnt/c/Users/dungbt06/Fine-tune
R=results/ft_strip
mkdir -p $R/gates
LOG=$R/gates.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
# G1: chấm lại adapter đối chứng trên dev_v1, batch 1, đúng tham số cũ; so từng byte
timeout 3600 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3-1.7B $R/gates/g1 --name Qwen3-1.7B-fullvocab-s1 \
  --adapter results/ft_loss/train/Qwen3-1.7B-s1/adapter --rows data/bench_v1/dev_v1.jsonl > $R/gates/g1.log 2>&1
say "G1 eval exit=$?"
if cmp -s results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl $R/gates/g1/Qwen3-1.7B-fullvocab-s1/preds.jsonl; then say "G1 PASS byte-identical"; else say "G1 FAIL"; fi
sha256sum results/dev_v1/eval/Qwen3-1.7B-fullvocab-s1/preds.jsonl $R/gates/g1/Qwen3-1.7B-fullvocab-s1/preds.jsonl | tee -a "$LOG"
# G2: đường dữ liệu 200 bước đầu — tham chiếu (code cũ) / mới strip 0 / mới strip 0.3
timeout 1200 bash setup/run.sh scripts/ft_strip/ref_batches.py 200 $R/gates/ref_batches.jsonl > $R/gates/g2_ref.log 2>&1; say "G2 ref exit=$? $(tail -1 $R/gates/g2_ref.log)"
for p in 0 0.3; do
  timeout 1200 bash setup/run.sh scripts/train_lora.py Qwen/Qwen3-1.7B 1 $R/gates/dump_p$p --loss full_vocab --strip_p $p \
    --strip_state_only_frac 0.5 --dump_batches 200 > $R/gates/g2_p$p.log 2>&1
  say "G2 dump p=$p exit=$? $(tail -1 $R/gates/g2_p$p.log)"
done
bash setup/run.sh scripts/ft_strip/compare_dumps.py >> "$LOG" 2>&1; say "G2 compare exit=$?"
# G3
timeout 600 bash setup/run.sh scripts/ft_strip/g3_strip_check.py > $R/gates/g3.log 2>&1; say "G3 exit=$?"
say GATES_DONE
