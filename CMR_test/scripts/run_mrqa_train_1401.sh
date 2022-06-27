# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

task="mrqa_squad"

# modelsize="large"
# lr=1e-5
# train_bsz=8
# pred_bsz=32
 
modelsize="base"
lr=5e-5
train_bsz=24
pred_bsz=24
num_epochs=30
output_dir="out/${task}_bart-${modelsize}_upstream_model"

warmup=100
max_input_length=888

gpu="1,2,3,4"

logname="train_bart-${modelsize}"

if [[ -f "/data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl" ]]
then
    echo "/data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl exists on your filesystem."
else
    shuf -n 2048 "/data2/home/gangwei/project/CMR/data/${task}/${task}_dev.jsonl" > "/data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl"
    echo "/data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl generated."
fi 

logfile=logs/${task}.${logname}.log
CUDA_VISIBLE_DEVICES=$gpu python cli_bart.py \
        --do_train \
        --output_dir ${output_dir} \
        --model facebook/bart-${modelsize} \
        --dataset mrqa \
        --train_file /data2/home/gangwei/project/CMR/data/${task}/${task}_train.jsonl \
        --dev_file /data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl \
        --test_file /data2/home/gangwei/project/CMR/data/${task}/${task}_dev.jsonl \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 1500 \
        --num_train_epochs ${num_epochs} \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  
        # > ${logfile}  2>&1 &

# tail -f logs/${task}.${logname}.log

#  --train_file /data2/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl \
echo "${logfile}"