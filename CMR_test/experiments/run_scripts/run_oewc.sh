#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# source ~/.bashrc
# conda activate cmr
# cd ~/CMR/

## Args ##
seed=$1
lr=$2
ep=$3
ewc_lambda=$4
ewc_gamma=$5

## Paths ##
ns_config=$6
task_name=$7
stream_split=$8
stream_id=$9


if [ "$task_name" = "qa" ]; then
    upstream_data_path="/data2/home/gangwei/project/CMR/data/mrqa_squad/mrqa_squad_train.jsonl"
    submission_stream_data="experiments/eval_data/qa/submission_stream.${ns_config}-${stream_split}.json"
    upstream_eval_data="experiments/eval_data/qa/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/qa/heldout_eval.jsonl"
    base_model_path="out/mrqa_squad_bart-base_upstream_model/best-model.pt"
    task_name_arg="mrqa"
elif [ "$task_name" = "nli" ]; then
    upstream_data_path="data/snli/snli_train.jsonl"
    submission_stream_data="experiments/eval_data/nli/submission_stream.${ns_config}-${stream_split}.json"
    upstream_eval_data="experiments/eval_data/nli/upstream_eval.jsonl"
    heldout_submission_data="experiments/eval_data/nli/heldout_eval.jsonl"
    base_model_path="out/snli_bart-base_1109_upstream_model/best-model.pt"
    task_name_arg="nli"
fi

if [ "$stream_split" = "val" ]; then
    use_wandb=False
    max_timecode=100
    save_ckpt_freq=100
    kr_eval_freq=50
    kg_eval_freq=50
elif [ "$stream_split" = "test" ]; then
    use_wandb=False
    max_timecode=100
    save_ckpt_freq=25
    kr_eval_freq=10
    kg_eval_freq=10
fi

gpu=1

prefix="${task_name}_oewc_lr=${lr}_ep=${ep}_lbd=${ewc_lambda}_gm=${ewc_gamma}_${ns_config}-${stream_split}[${stream_id}]_seed=${seed}"
ckpt_dir="experiments/ckpt_dirs/${task_name}/owc/${prefix}"
mkdir -p ${ckpt_dir}


log_file="experiments/logs/run_1110_${prefix}.log"
echo "Starting ${log_file}."
touch ${log_file} 

CUDA_VISIBLE_DEVICES=$gpu python cmr/debug_algs/run_lifelong_finetune.py \
    --use_wandb ${use_wandb} \
    --seed $seed --stream_id ${stream_id} \
    --cl_method "online_ewc" \
    --ewc_lambda ${ewc_lambda} --ewc_gamma ${ewc_gamma} \
    --learning_rate ${lr} --num_train_epochs ${ep} \
    --base_model_path ${base_model_path} \
    --num_beams 3 \
    --train_batch_size 2 \
    --predict_batch_size 2 \
    --max_timecode ${max_timecode} \
    --kr_eval_freq ${kr_eval_freq} --kr_eval_mode "metric" \
    --kg_eval_freq ${kg_eval_freq} --kg_eval_mode "metric" \
    --prefix ${prefix} \
    --submission_stream_data ${submission_stream_data} \
    --upstream_eval_data ${upstream_eval_data} \
    --heldout_submission_data ${heldout_submission_data} \
    --save_ckpt_freq ${save_ckpt_freq} \
    --ckpt_dir ${ckpt_dir} \
    --result_file "experiments/results/${task_name}/${prefix}_result.json" > ${log_file} 
    # 2>&1 
    # &
# tail -f ${log_file}
echo "Finished ${log_file}."
exit
# exit
# exit


