import argparse
import logging
import os
import numpy as np
import torch

from options.base_options import BaseOptions
from utils import utils
from models import create_model
from tqdm import tqdm, trange
from datasets import load_labeled_dataset, load_aux_dataset
from transformers import (
	MODEL_WITH_LM_HEAD_MAPPING,
	WEIGHTS_NAME,
	AdamW,
	PreTrainedModel,
	PreTrainedTokenizer,
	get_linear_schedule_with_warmup,
)

from models.aux_process import run_aux_batch
logger = logging.getLogger(__name__)


def process_task_batch(model, batch, tokenizer, args, task_name, pri_model=None):
    if task_name in ["mlm", "lm"]:
        output = run_aux_batch(model, batch, tokenizer, args, task_name)
        loss = output[0]
    else:
        raise ValueError("unrecognzied task named %s !" % task_name)        
    return loss


def train(args, train_dataset,  model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    model.resize_token_embeddings(len(tokenizer))
    accum_dataset_len, t_total, task_lens = 0, 0, []
    for task in args.task_order:
        if task[0]=="a":
            dataset = train_dataset[args.aux_task_names[int(task[1])]]  
        else:
            continue
        accum_dataset_len += len(dataset)
        t_total += len(dataset) // (args.per_gpu_train_batch_size)
        if args.task_mix:
            if task[0]=="a":
                task_lens.append(t_total)
            else:
                task_lens[-1] = t_total
        else:
            task_lens.append(t_total)
        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         "weight_decay": 0.0},
    ]

    # Setup the optimizer for the base model
    optimizer = AdamW(optimizer_grouped_parameters, betas=eval(args.betas),
                        lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.base_wd)
    scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=int(args.warmup_frac * t_total), num_training_steps=t_total)

    logger.info("***** Running training *****")
    for k, v in train_dataset.items():
        logger.info("Aux Task= {} Num examples = {}".format(k, len(v)))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num Warmup Steps = %d", int(args.warmup_frac * t_total))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = {}. Will eval every {}".format(t_total, args.eval_every))

    current_task_index = -1    
    current_domain_index = -1
    model.zero_grad()
    
    for step in tqdm(range(t_total)):        
        # if args.task_mix:
        if step >= task_lens[current_domain_index] or step == 0:
            datas = []
            data_loaders = []
            for _ in range(len(args.task_order)):
                current_task_index += 1
                task = args.task_order[current_task_index]
                data = train_dataset[args.aux_task_names[int(task[1])]] if task[0]=="a" else primary_dataset[args.train_tasks[int(task[1])]]
                datas.append(data)
                data_loaders.append(iter(data.load_data()))
                if (not args.task_mix) or current_task_index+1==len(args.task_order) or args.task_order[current_task_index+1][0]=="a":
                    break
            current_domain_index += 1
        # else:
        #     if step >= task_lens[current_task_index] or step == 0:
        #         current_task_index += 1
        #         task = args.task_order[current_task_index]
        #         data = train_dataset[args.aux_task_names[int(task[1])]] if task[0]=="a" else primary_dataset[args.train_tasks[int(task[1])]]
        #         datas = [data]
        #         data_loaders = [iter(dataset.load_data())]
        #     return
        try:
            used_task_idx = np.random.randint(0, len(datas))
            task_data = datas[used_task_idx]
            batch = next(data_loaders[used_task_idx])
        except:
            print("The train data %s is used out!" % task_data.task_name)
            del datas[used_task_idx]
            del data_loaders[used_task_idx]
            used_task_idx = np.random.randint(0, len(datas))
            task_data = datas[used_task_idx]
            batch = next(data_loaders[used_task_idx])
            
        loss = process_task_batch(model, batch, tokenizer, args, task_data.task_name)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()

        if step % args.eval_every == 0:
            print("eval...")
            

        

def main():
    args = BaseOptions().parse()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Set seed
    utils.set_seed(args)

    model_name = args.model_name_or_path
    assert model_name, 'The name of the model is not Set. Maybe use roberta-base as the default'

    model, _, tokenizer = create_model(args)

    # Setting up the dataset
    aux_dataset = load_aux_dataset(args, tokenizer)
    
    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)
	
    model.to(args.device)
    
    train(args, aux_dataset, model, tokenizer)


if __name__ == "__main__":
    main()
