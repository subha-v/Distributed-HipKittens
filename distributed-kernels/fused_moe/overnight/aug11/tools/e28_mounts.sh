#!/usr/bin/env bash
set -uo pipefail
echo "===MOUNTS==="
docker inspect subha_k1 --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{"\n"}}{{end}}'
echo "===LOOK==="
docker exec subha_k1 bash -lc 'ls -d /home/subvadla /workspace /root 2>&1 | head; ls /home 2>&1 | head'
echo "===FIND_DONOR_IN_CONTAINER==="
docker exec subha_k1 bash -lc 'ls -d /home/subvadla/amd-master 2>&1; ls -d /home/subvadla/Distributed-HipKittens 2>&1'
echo "===DONE==="
