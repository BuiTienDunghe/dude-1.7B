#!/usr/bin/env bash
which -a python3 python3.12 python3.13 python3.14 2>/dev/null
python3 -c 'import venv, ensurepip; print("venv ok")'
nproc; free -g; df -h /mnt/c "$HOME" | tail -2
cat /etc/os-release | head -2
which uv curl git
