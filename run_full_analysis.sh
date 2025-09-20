#!/bin/bash
if [ -z "$2" ]; then
  echo "Usage: $0 <model_name> <dataset> [conda_env] [where_model_path]"
  exit 1
fi

MODEL_NAME=$1
DATASET=$2
ENV_NAME=${3:-condaenv}
WHERE_MODEL_PATH=${4:-$MODEL_NAME}

# Sanitize model name for log file by replacing slashes
LOG_MODEL_NAME=${MODEL_NAME//\//_}
LOG_FILE="paper/analysis_${LOG_MODEL_NAME}_${DATASET}.txt"

# Create directory for log file if it doesn't exist
mkdir -p paper

# Redirect all output (stdout and stderr) to both console and log file
{
    SOURCE_TABLE_NAME="reasoning_traces_${DATASET}"

    eval "$(conda shell.bash hook)"

    conda activate "$ENV_NAME"

    if [ $? -ne 0 ]; then
    echo "Error: Conda environment '$ENV_NAME' not found or could not be activated."
    exit 1
    fi

    python analyze_accuracy_vs_length.py --model_name "$MODEL_NAME" --where_model_path "$WHERE_MODEL_PATH" --source_table_name "$SOURCE_TABLE_NAME"

    echo "========================================"

    python analyze_critical_chunk_prob_change.py --model_name "$MODEL_NAME" --where_model_path "$WHERE_MODEL_PATH" --source_table_name "$SOURCE_TABLE_NAME"

    echo "========================================"

    python analyze_probability_jump_heuristic.py --model_name "$MODEL_NAME" --where_model_path "$WHERE_MODEL_PATH" --source_table_name "$SOURCE_TABLE_NAME"

} 2>&1 | tee "${LOG_FILE}"
