#!/bin/bash
# Waits until any of the given kernels (rabeya100x/<slug>) is no longer running; prints all statuses.
cd "$(dirname "$0")/.."
while true; do
  done_any=0; out=""
  for k in "$@"; do
    s=$(venv/bin/kaggle kernels status "rabeya100x/$k" 2>&1 | grep -oE "COMPLETE|ERROR|RUNNING|QUEUED|CANCEL[A-Z_]*" | head -1)
    out="$out$k=$s "
    if [[ "$s" == "COMPLETE" || "$s" == "ERROR" || "$s" == CANCEL* ]]; then done_any=1; fi
  done
  if [[ $done_any == 1 ]]; then echo "$(date -u) $out"; exit 0; fi
  sleep 120
done
