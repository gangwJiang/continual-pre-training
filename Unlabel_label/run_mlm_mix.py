import argparse
import logging
import os
import numpy as np
import torch

from options.base_options import BaseOptions
from utils import utils
from models import create_model, create_pri_model
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
from models.seq_classification import run_seq_cla_batch
from models.token_classification import run_token_cla_batch
from models.question_answer import run_qa_batch

logger = logging.getLogger(__name__)



def train(args, train_dataset, primary_dataset, model: PreTrainedModel, pri_model, tokenizer: PreTrainedTokenizer, test_dataset, valid_dataset):
    model.resize_token_embeddings(len(tokenizer))
    accum_dataset_len, t_total, task_lens = 0, 0, []
    for task in args.task_order:
        dataset = train_dataset[args.aux_task_names[int(task[1])]] if task[0]=="a" else primary_dataset[args.train_tasks[int(task[1])]]
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
    optimizer_extend = AdamW(pri_model.parameters(), betas=eval(args.betas),
                        lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.base_wd)
    
    scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=int(args.warmup_frac * t_total), num_training_steps=t_total)
    scheduler_extend = get_linear_schedule_with_warmup(
                    optimizer_extend, num_warmup_steps=int(args.warmup_frac * t_total), num_training_steps=t_total)
    

    logger.info("***** Running training *****")
    for k, v in train_dataset.items():
        logger.info("Aux Task= {} Num examples = {}".format(k, len(v)))
    for k, v in primary_dataset.items():
        logger.info("Primary Task= {} Num examples = {}".format(k, len(v)))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num Warmup Steps = %d", int(args.warmup_frac * t_total))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = {}. Will eval every {}".format(t_total, args.eval_every))

    current_task_index = -1    
    current_domain_index = -1
    model.zero_grad()
    pri_model.zero_grad()
    
    for _ in range(1):
        current_task_index = -1    
        current_domain_index = -1
    # for step in tqdm(range(t_total)): 
        for step in range(t_total): 
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
            
            loss = process_task_batch(model, pri_model, batch, tokenizer, args, task_data.task_name)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            
            if args.task_order[used_task_idx][0]=="d":
                torch.nn.utils.clip_grad_norm_(pri_model.parameters(), args.max_grad_norm)
                optimizer_extend.step()
                scheduler_extend.step()
                pri_model.zero_grad()

            if step % 100 == 0:
                logger.info("step %d : loss %f ." % (step, loss))
            
            if step % args.eval_every == 0:
                logger.info(">>> eval...")
                for i in range(len(args.train_tasks)):
                    evaluate(args, 'Test', test_dataset[args.train_tasks[i]], args.train_tasks[i], model, pri_model)
                    evaluate(args, 'Valid', valid_dataset[args.train_tasks[i]], args.train_tasks[i], model, pri_model)
        
            
# Evaluate the classifier
def evaluate(args, flag, dataset, name, model, pri_model):
    torch.cuda.empty_cache()
    # reset the metrics before running new stuff
    # Run the classifier
    pri_model.eval()
    model.eval()
    
    total_loss=0
    total_acc=0
    total_num=0
    target_list = []
    pred_list = []
    with torch.no_grad():
        for batch in dataset.load_data():
            inputs, segment_ids, input_mask, labels = batch	
            real_b = inputs.size(0)
            inputs = inputs.to(args.device)
            segment_ids = segment_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            labels = labels.to(args.device)
            output = pri_model(model.roberta, inputs, token_type_ids=segment_ids, attention_mask=input_mask, labels=labels)
            loss = output[0]
            _, pred = output[1].max(1)
            # print(pred, labels)
            hits = sum(list(pred==labels))
            target_list.append(labels)
            pred_list.append(pred)
            # Log
            total_loss+=loss.data.cpu().numpy().item() * real_b
            total_acc+=hits
            total_num+=real_b
        test_loss, test_acc = total_loss/total_num, total_acc/total_num   
        # f1 = utils.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')
    
    logger.info('>>> {:5s} on task {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(flag, name, total_loss/total_num, 100*total_acc/total_num))

    pri_model.train()
    model.train()
    # Get the metrics from the classifier
    torch.cuda.empty_cache()
    return test_loss, test_acc


def process_task_batch(model, pri_model, batch, tokenizer, args, task_name):
    if task_name in ["mlm", "lm"]:
        output = run_aux_batch(model, batch, tokenizer, args, task_name)
        loss = output[0]
    elif task_name in ["seq_cla"]:
        output = run_seq_cla_batch(model, pri_model, batch, args)
        loss = output[0]
    elif task_name in ["token_cla"]:
        output = run_token_cla_batch(model, pri_model, batch, args)
        loss = output[0]
    elif task_name in ["qa"]:
        output = run_qa_batch(model, pri_model, batch, args)
        loss = output[0]
    else:
        raise ValueError("unrecognzied task named %s !" % task_name)        
    return loss


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

    model, tokenizer = create_model(args)

    # Setting up the dataset
    aux_dataset = load_aux_dataset(args, tokenizer)
    primary_dataset, num_labels = load_labeled_dataset(args, tokenizer)
    primary_test_dataset, _ = load_labeled_dataset(args, tokenizer, "test")
    primary_val_dataset, _ = load_labeled_dataset(args, tokenizer, "valid")
        
    pri_model = create_pri_model(args, num_labels)
    
    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)
	
    model.to(args.device)
    pri_model.to(args.device)
    
    train(args, aux_dataset, primary_dataset, model, pri_model, tokenizer, primary_test_dataset, primary_val_dataset)


if __name__ == "__main__":
    main()
