#!/bin/bash

# ==================================
# Config
# ==================================
export CUDA_VISIBLE_DEVICES=1
DATASETS=("RFNet" "NTUHumanID" "NTUHAR" "SignFi" "Baha" "Xrf55" "WiCount" "WiFallact" "WiGestureact" "WiFallid" "WiGestureid")
MODELS=("MLP" "ResNet18" "ResNet50" "ResNet101" "RNN" "GRU" "LSTM" "BiLSTM" "Transformer")

data_root="/home/chenjiayi/workspace/willm/wifi_data"

BASE_OUTPUT_DIR="baseline"
GLOBAL_SUMMARY="${BASE_OUTPUT_DIR}/summary_all.txt"

# 🔥 设置并行度（你可以改这个）
PARALLEL=4

mkdir -p $BASE_OUTPUT_DIR

# ==================================
# 初始化全局 summary
# ==================================

echo "=================================================" | tee $GLOBAL_SUMMARY
echo "Global Baseline Summary" | tee -a $GLOBAL_SUMMARY
echo "Start Time: $(date)" | tee -a $GLOBAL_SUMMARY
echo "Parallel Workers: $PARALLEL" | tee -a $GLOBAL_SUMMARY
echo "=================================================" | tee -a $GLOBAL_SUMMARY
echo "" | tee -a $GLOBAL_SUMMARY
echo "Dataset | Model | Best Accuracy | Time(s)" | tee -a $GLOBAL_SUMMARY
echo "-------------------------------------------------" | tee -a $GLOBAL_SUMMARY

# ==================================
# 并发控制函数
# ==================================

function wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
        sleep 1
    done
}

# ==================================
# 主循环
# ==================================

for DATASET in "${DATASETS[@]}"; do

    DATASET_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DATASET}"
    SUMMARY_FILE="${DATASET_OUTPUT_DIR}/summary.txt"
    mkdir -p $DATASET_OUTPUT_DIR

    echo "=========================================" > $SUMMARY_FILE
    echo "Dataset: $DATASET" >> $SUMMARY_FILE
    echo "Start Time: $(date)" >> $SUMMARY_FILE
    echo "=========================================" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE

    for MODEL in "${MODELS[@]}"; do

        wait_for_slot   # 🔥 控制并发

        (
            OUTPUT_PATH="${DATASET_OUTPUT_DIR}/${MODEL}_output"
            mkdir -p $OUTPUT_PATH

            echo "Running $MODEL on $DATASET"

            START_TIME=$(date +%s)

            python single_dataset_baseline.py \
                --model $MODEL \
                --output_path $OUTPUT_PATH \
                --data_root $data_root \
                --dataset $DATASET \
                --train_ratio 1.0

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            LOG_FILE=$(find $OUTPUT_PATH -name "log.txt" | head -n 1)

            if [ -f "$LOG_FILE" ]; then
                BEST_LINE=$(grep "Best Test Accuracy" $LOG_FILE)
                BEST_ACC=$(echo $BEST_LINE | awk '{print $4}')
            else
                BEST_ACC="N/A"
            fi

            # 用 flock 防止并发写冲突
            (
                flock -x 200

                echo "Model: $MODEL" >> $SUMMARY_FILE
                echo "Best Accuracy: $BEST_ACC" >> $SUMMARY_FILE
                echo "Training Time: ${DURATION}s" >> $SUMMARY_FILE
                echo "----------------------------------------" >> $SUMMARY_FILE

                echo "$DATASET | $MODEL | $BEST_ACC | $DURATION" >> $GLOBAL_SUMMARY

            ) 200>>${BASE_OUTPUT_DIR}/.lockfile

        ) &   # 🔥 后台运行

    done
done

# 等待所有任务结束
wait

echo "" | tee -a $GLOBAL_SUMMARY
echo "All Experiments Finished at: $(date)" | tee -a $GLOBAL_SUMMARY
echo "=================================================" | tee -a $GLOBAL_SUMMARY