#!/usr/bin/env bash
cd /mnt/c/Users/dungbt06/Fine-tune
for v in .venv .venv-baselines; do
  echo "== $v"
  $v/bin/python -c "
import importlib
for m in ['torch','transformers','sentencepiece','tokenizers','protobuf','google.protobuf','huggingface_hub','datasets']:
    try:
        x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m,'MISSING',type(e).__name__)
" 2>&1 | tail -n 9
done
