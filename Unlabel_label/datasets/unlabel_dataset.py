import os
import pickle
import logging

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from utils.utils import collate

logger = logging.getLogger(__name__)


map_data_dir = {
	"bio": "/data2/home/gangwei/project/dataset/continual/datasets/chemprot/domain.10xTAPT.txt",
	"cs": "/data2/home/gangwei/project/dataset/continual/datasets/sciie/domain.10xTAPT.txt" 
}


'''
	Dataset wrangling related code
'''
class TextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, task_name,  block_size=512):
		file_path = map_data_dir[task_name]
  
		assert os.path.isfile(file_path)

		block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

		directory, filename = os.path.split(file_path)
		cached_features_file = os.path.join(
			directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
		)

		if os.path.exists(cached_features_file) and not args.overwrite_cache:
			logger.info("Loading features from cached file %s", cached_features_file)
			with open(cached_features_file, "rb") as handle:
				self.examples = pickle.load(handle)
		else:
			logger.info("Creating features from dataset file at %s", directory)

			self.examples = []
			with open(file_path, encoding="utf-8") as f:
				text = f.read()

			tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

			for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
				self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
			# Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
			# If your dataset is small, first you should loook for a bigger one :-) and second you
			# can change this behavior by adding (model specific) padding.

			logger.info("Saving features into cached file %s", cached_features_file)
			with open(cached_features_file, "wb") as handle:
				pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, item):
		return torch.tensor(self.examples[item], dtype=torch.long)


def get_tokenized_file(file_path:str, tokenizer: PreTrainedTokenizer, block_size=512, shuffle=False, lazy=False):
	logger.info("Creating features from dataset file at %s", file_path)
	logger.info("Reading Line by Line")
	lines = []
	with open(file_path, encoding="utf-8") as f:
		for line in f:
			if len(line) > 0 and not line.isspace():
				lines.append(line)
	logger.info("Done Reading Line By Line. About to pass through the tokenize")
	if lazy:
		return lines
	return tokenizer.batch_encode_plus(lines, truncation=True, add_special_tokens=True, max_length=block_size)["input_ids"]


class LineByLineTextDataset(Dataset):
	def __init__(self, tokenizer: PreTrainedTokenizer, args, task_name, lazy:bool, block_size=512):
		file_path = map_data_dir[task_name]
  
		assert os.path.isfile(file_path)
		# Here, we do not cache the features, operating under the assumption
		# that we will soon use fast multithreaded tokenizers from the
		# `tokenizers` repo everywhere =)
		self.lazy = lazy
		self.block_size = block_size

		self.tokenizer = tokenizer
		self.examples = get_tokenized_file(file_path, tokenizer, block_size, lazy=lazy, logger=logger)


	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		tokenized = self.examples[i]
		if self.lazy:
			tokenized = self.tokenizer.encode_plus(tokenized, truncation=True, add_special_tokens=True, max_length=self.block_size)["input_ids"]
		
		return collate(torch.tensor(tokenized, dtype=torch.long), self.tokenizer.pad_token_id)