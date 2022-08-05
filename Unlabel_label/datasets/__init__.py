import torch
import json
from datasets.unlabel_dataset import LineByLineTextDataset, TextDataset
from utils.utils import collate_fn



def load_aux_dataset(args, tokenizer):
    task_names = args.aux_task_names
    task_types = args.aux_task_types
    assert len(task_names) == len(task_types), 'Mismatch between the number of train files for MLM and the number of aux task names'
    datasets = {}
    for idx, task_name in enumerate(task_names):
        if task_name not in ["cs", "bio"]:
            raise ValueError("No such task in axu dataset %s !!!" % task_name)
        if args.line_by_line:
            dataset = LineByLineTextDataset(tokenizer, args, task_name, lazy=args.lazy_dataset, block_size=args.block_size)
        else:
            dataset = TextDataset(tokenizer, args, task_name, block_size=args.block_size)
        datasets[task_name] = CustomDatasetDataLoader(args, dataset, args.per_gpu_train_batch_size, task_types[idx])
    return datasets



def load_labeled_dataset(args, tokenizer, train_type="train"):
    task_names = args.train_tasks
    num_labels = 0
    datasets = {}
    for idx, task_name in enumerate(task_names):
        if task_name in ['XuSemEval14_rest', 'XuSemEval14_laptop']:
            from datasets.cla_dataset import ASClassificationDataset
            dataset = ASClassificationDataset(tokenizer, task_name, args.max_seq_len, train_type, lazy=False)
        elif task_name in ['chemprot', 'citation_intent', 'partisan']:
            from datasets.cla_dataset import SciClassificationDataset
            dataset = SciClassificationDataset(tokenizer, task_name, args.max_seq_len, train_type, lazy=False)
        else:
            raise ValueError("No such task %s !!!" % task_name)
            # datasets[task_name] = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
        if len(dataset.id2label)>num_labels:
            num_labels = len(dataset.id2label)
        if train_type=="train":
            datasets[task_name] = CustomDatasetDataLoader(args, dataset, args.per_gpu_train_batch_size, args.pri_task)
        else:
            datasets[task_name] = CustomDatasetDataLoader(args, dataset, args.per_gpu_eval_batch_size, args.pri_task)
    return datasets, num_labels





class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, args, dataset, bsz, task_name=None):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.args = args
        self.dataset_len = len(dataset)
        self.task_name = task_name
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bsz,
            # collate_fn=collate_fn(tokenizer.pad_token_id), 
            drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_len
