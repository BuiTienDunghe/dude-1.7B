#!/usr/bin/env bash
# Bench 1.3 DỰNG THỬ, bước 3: chấm lại các dòng dev mới/đổi (rescore/rescore_dev.jsonl) bằng 9 model: 7 model của scores_v1_2.json, Qwen3.5-2B gốc
# (eval_base, n_tokens để trống như bước E), Quyet-1.0-Medium (adapter bước E, .venv-baselines, OFFLINE). Batch 1. CHỈ DEV; không test, không train.
cd /mnt/c/Users/dungbt06/Fine-tune
T=results/bench_v1_3_trial; RS=$T/rescore/rescore_dev.jsonl; E=$T/eval_rescore
bash scripts/human_edits/rescore_rows_v1_2.sh $RS $E
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -c "import json; rows=[json.loads(l) for l in open('$RS',encoding='utf-8')]; open('$T/rescore/rescore_dev_ntok_null.jsonl','w',encoding='utf-8').write(''.join(json.dumps({**r,'n_tokens':None},ensure_ascii=False)+'\n' for r in rows))"
timeout 1800 bash setup/run.sh -X faulthandler scripts/eval_base.py Qwen/Qwen3.5-2B $E --name Qwen3.5-2B --rows $T/rescore/rescore_dev_ntok_null.jsonl > $E/log_qwen35.txt 2>&1
echo "qwen35 exit=$?"
timeout 1800 .venv-baselines/bin/python -u -X faulthandler scripts/bench_v1_3/score_rows_ext.py quyet-medium $RS $E/ext > $E/log_quyet_medium.txt 2>&1
echo "quyet-medium exit=$?"
echo SCORE_V1_3_DONE
