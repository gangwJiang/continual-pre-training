# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

task="mrqa_squad"

# modelsize="large"
# lr=1e-5
# train_bsz=8
# pred_bsz=32
 
modelsize="base"
lr=5e-5
train_bsz=4
pred_bsz=4
num_epochs=30
output_dir="out/${task}_bart-${modelsize}_upstream_model"

warmup=100
max_input_length=888
 

logname="train_bart-${modelsize}"

if [[ -f "/amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl" ]]
then
    echo "/amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl exists on your filesystem."
else
    shuf -n 2048 "/amax/home/gangwei/project/CMR/data/${task}/${task}_dev.jsonl" > "/amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl"
    echo "/amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl generated."
fi 

logfile=logs/${task}.${logname}.log
python cli_bart.py \
        --do_train \
        --output_dir ${output_dir} \
        --model facebook/bart-${modelsize} \
        --dataset mrqa \
        --train_file /amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl \
        --dev_file /amax/home/gangwei/project/CMR/data/${task}/${task}_dev.mini.2048.jsonl \
        --test_file /amax/home/gangwei/project/CMR/data/${task}/${task}_dev.jsonl \
        --learning_rate ${lr} \
        --warmup_steps ${warmup} \
        --train_batch_size ${train_bsz} \
        --predict_batch_size ${pred_bsz} \
        --eval_period 10 \
        --num_train_epochs ${num_epochs} \
        --max_input_length ${max_input_length} \
        --max_output_length 50 \
        --num_beams 3 \
        --append_another_bos  
        # > ${logfile}  2>&1 &

# tail -f logs/${task}.${logname}.log

# --train_file /amax/home/gangwei/project/CMR/data/${task}/${task}_train.jsonl \
echo "${logfile}"