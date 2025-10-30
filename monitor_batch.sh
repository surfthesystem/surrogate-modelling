#!/bin/bash
while true; do
  echo "=== Progress Check $(date '+%H:%M:%S') ==="
  
  # Count active simulations
  active=$(ps aux | grep "IMPES_phase1.py" | grep -v grep | wc -l)
  echo "Active IMPES simulations: $active"
  
  # Count completed scenarios
  completed=$(find results/training_data -type d -name "doe_*" 2>/dev/null | wc -l)
  echo "Completed scenarios: $completed/10"
  
  # Check batch log for completion messages
  tail -50 batch_10scenarios.log | grep -E "^(✓|✗)" | tail -5
  
  # Check if batch process is still running
  batch_running=$(ps aux | grep "batch_simulator.*max_scenarios 10" | grep -v grep | wc -l)
  if [ "$batch_running" -eq 0 ]; then
    echo "Batch process completed!"
    break
  fi
  
  echo ""
  sleep 120  # Check every 2 minutes
done
