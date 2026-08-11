#!/usr/bin/env bash
# Cheap "is a GPU job of ours running?" probe. Safe to run at any time; reads
# only. Used to decide whether the next experiment may take the GPU.
set -u
echo "kfd_pids: $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/{print $1}' | wc -l)"
rocm-smi --showpids 2>&1 | sed -n '/PID/,/^====/p'
echo "-- util --"
rocm-smi 2>&1 | tail -12 | awk 'NR>2{printf "GPU%s use=%s vram=%s pwr=%s sclk=%s\n",$1,$NF,$(NF-1),$5,$7}'
echo "-- our harness procs --"
docker exec dhk-gemmrs ps -eo pid,etime,cmd 2>/dev/null | grep -E 'python3|build\.sh|hipcc' | grep -v grep | head -20 || echo "  none"
