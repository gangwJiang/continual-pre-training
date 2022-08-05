
import logging
logger = logging.getLogger(__name__)

import os 
import torch
from datasets import data_utils
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


map_data_dir = {
	'XuSemEval14_rest': "/data2/home/gangwei/project/PyContinual-main/dat/absa/XuSemEval/asc/14/rest",
	'XuSemEval14_laptop': "/data2/home/gangwei/project/PyContinual-main/dat/absa/XuSemEval/asc/14/laptop",
    'chemprot': "/data2/home/gangwei/project/dataset/continual/datasets/chemprot",
    'citation_intent': "/data2/home/gangwei/project/dataset/continual/datasets/citation_intent",
    "partisan": "/data2/home/gangwei/project/dataset/continual/datasets/hyperpartisan"
}


class ClaProcessor(data_utils.DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['polarity']
            examples.append(
                data_utils.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ASClassificationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, task_name, max_seq_len, train_type, lazy=False):
        file_path = map_data_dir[task_name]    
          
        logger.info("Creating features from %s dataset file at %s", train_type, file_path)
        
        processor = ClaProcessor()
        label_list = processor.get_labels()
        
        if train_type == "train":
            train_examples = processor.get_train_examples(file_path)
        elif train_type == "valid":
            train_examples = processor.get_dev_examples(file_path)
        elif train_type == "test":
            train_examples = processor.get_test_examples(file_path)
        
        # print(tokenizer)
        self.train_features, self.id2label = data_utils.convert_examples_to_features(
            train_examples, label_list, max_seq_len, tokenizer, "asc")
        
    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, item):
        input_ids = torch.tensor(self.train_features[item].input_ids, dtype=torch.long)
        segment_ids = torch.tensor(self.train_features[item].segment_ids, dtype=torch.long)
        input_mask = torch.tensor(self.train_features[item].input_mask, dtype=torch.long)
        label_ids = torch.tensor(self.train_features[item].label_id, dtype=torch.long)
        # tasks = torch.tensor(t, dtype=torch.long)
        return input_ids, segment_ids, input_mask, label_ids


class SciProcessor(ClaProcessor):
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%d" % (set_type, i)
            text_a = lines[i]['text']
            label = lines[i]['label']
            examples.append(
                data_utils.InputExample(guid=guid, text_a=text_a, label=label))
        return examples



class SciClassificationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, task_name, max_seq_len, train_type, lazy=False):
        file_path = map_data_dir[task_name]    
          
        logger.info("Creating features from %s dataset file at %s", train_type, file_path)
        
        processor = SciProcessor()
        label_list = []
        
        if train_type == "train":
            train_examples = processor.get_train_examples(file_path, "train.jsonl")
        elif train_type == "valid":
            train_examples = processor.get_dev_examples(file_path, "dev.jsonl")
        elif train_type == "test":
            train_examples = processor.get_test_examples(file_path, "test.jsonl")
        
        # print(tokenizer)
        self.train_features, self.id2label = data_utils.convert_examples_to_features(
            train_examples, label_list, max_seq_len, tokenizer, "sci")
        
    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, item):
        input_ids = torch.tensor(self.train_features[item].input_ids, dtype=torch.long)
        segment_ids = torch.tensor(self.train_features[item].segment_ids, dtype=torch.long)
        input_mask = torch.tensor(self.train_features[item].input_mask, dtype=torch.long)
        label_ids = torch.tensor(self.train_features[item].label_id, dtype=torch.long)
        # tasks = torch.tensor(t, dtype=torch.long)
        return input_ids, segment_ids, input_mask, label_ids
