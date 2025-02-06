#!/bin/bash
count=100
while [ $count -gt 0 ]; do
    echo "Remaining runs: $count"
    python3 eval_ivis.py
    count=$((count-1))  # カウントダウン
done