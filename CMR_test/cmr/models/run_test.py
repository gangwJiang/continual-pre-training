# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This script was based on https://github.com/shmsw25/bart-closed-book-qa.
import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from cmr.task_manager.dataloader import GeneralDataset

from .mybart import MyBart
from .utils import freeze_embeds, trim_batch, convert_model_to_single_gpu
import json

from tqdm import tqdm
import copy


def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    train_data = GeneralDataset(logger, args, args.train_file,
                                data_type="train", is_training=True, task_name=args.dataset)
    dev_data = GeneralDataset(logger, args, args.dev_file,
                              data_type="dev", is_training=False, task_name=args.dataset)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    best_dev_performance = None
    test_performance = None

    best_model_state_dict = None

    if args.do_train:
        if args.checkpoint is not None and args.checkpoint != "None":
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_model_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.model)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        args.total_steps = args.num_train_epochs * len(train_data.dataloader)
        logger.info(f"args.total_steps = {args.total_steps}")
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
        best_dev_performance, best_model_state_dict = train(
            args, logger, model, train_data, dev_data, optimizer, scheduler)

    return best_dev_performance, test_performance


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_performance = None
    stop_training = False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet):
            global_step += 1
            if torch.cuda.is_available():
                # logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")
                batch = [b.to(torch.device("cuda")) for b in batch]

            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()

                for batch in tqdm(dev_data.dataloader, desc="Infernece", disable=dev_data.args.quiet):
                    if torch.cuda.is_available():
                        batch = [b.to(torch.device("cuda")) for b in batch]
                    pad_token_id = dev_data.tokenizer.pad_token_id
                    batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
                    outputs = model.module.generate(input_ids=batch[0],
                                    attention_mask=batch[1],
                                    num_beams=dev_data.args.num_beams,
                                    max_length=dev_data.args.max_output_length,)
                    for input_, output in zip(batch[0], outputs):
                        pred = dev_data.decode(output)
                        print(pred)
                model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break

    # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    # torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))
    return best_performance
