#!/bin/bash

NUM_CORES=8

for ((i=1; i<=$NUM_CORES; i++))
do
  SEED=$i
  OUTPUT_FILE="Even_Sparser_ICM_F_${i}"
  MAX_MODEL_RUNS=1
  MAX_TRAINING_EPISODES=50000
  # Record the start time
  start_time=$(date +%s)

  nohup python3 -u train_ICM.py --seed $SEED --output "$OUTPUT_FILE" --max_model_runs $MAX_MODEL_RUNS --max_training_episodes $MAX_TRAINING_EPISODES > "nohup/nohup_vm/${OUTPUT_FILE}.log" 2>&1 &
  disown

  # Wait for the process to complete
  wait
  # Record the end time
  end_time=$(date +%s)

  # Calculate the time difference
  elapsed_time=$((end_time - start_time))

  # Append the elapsed time to the log file
  echo "Iteration ${i} took ${elapsed_time} seconds" >> "nohup/nohup_vm/${OUTPUT_FILE}.log"
done

