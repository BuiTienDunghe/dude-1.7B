#!/usr/bin/env bash
for d in ~/.cache/huggingface/hub /mnt/c/Users/dungbt06/.cache/huggingface/hub; do
  ls -d $d/datasets--uitnlp--vihsd 2>/dev/null && find $d/datasets--uitnlp--vihsd -maxdepth 4 -name "*.md" 2>/dev/null
done
