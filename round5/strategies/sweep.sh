#!/bin/bash
cd /Users/alt2005/IMC/kernel_trick/ROUND5/strategies
export PYTHONPATH=/Users/alt2005/IMC/kernel_trick/imc-prosperity-4-backtester
export SHOCK_THRESH=0.005 ENTRY_SIZE=10 MAX_HOLD_TICKS=20 MM_HALF_SPREAD=2 MM_QUOTE_SIZE=1

# Train days 2-3, hold out day 4
printf "%-6s %-10s %-10s %s\n" "skew" "day2" "day3" "sum"

for skew in 1 2 3 4 5 6 7 8 9 10; do
  out=$(MM_MAX_SKEW=$skew \
    python -m prosperity4bt robot_trader.py 5-2 5-3 --no-vis --no-progress --no-out 2>&1 \
    | grep "^Total profit:" | awk '{gsub(",", "", $3); print $3}')
  d2=$(echo "$out" | sed -n '1p')
  d3=$(echo "$out" | sed -n '2p')
  sum=$((d2 + d3))
  printf "%-6s %-10s %-10s %s\n" "$skew" "$d2" "$d3" "$sum"
done
