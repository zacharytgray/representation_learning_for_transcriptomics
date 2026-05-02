#!/usr/bin/env bash
# all-geneset replication launcher; resumable via per-cell CSV checkpoint
set -euo pipefail
cd /home/zgray/Github/representation_learning_for_transcriptomics
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
log="logs/headline_${ts}.log"
echo "[$(date)] starting headline run -> $log"
exec .venv/bin/python -m src.replication --geneset all >"$log" 2>&1
