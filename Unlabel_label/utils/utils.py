from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer
import os
import os
import glob
import re
from typing import Dict, List, Tuple

import random
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

# For collating text data into batches and padding appropriately
def collate(examples, pad_token_id):
	return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)

def collate_fn(pad_token_id):
	def this_collate(examples):
		return collate(examples, pad_token_id)
	return this_collate


def f1_compute_fn(y_true, y_pred, average):
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred,average=average)


'''
	Misc setup code. 
'''
def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)
  

'''
	Checkpointing related code. Inherited from Huggingface.
'''
def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
	ordering_and_checkpoint_path = []
	glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
	for path in glob_checkpoints:
		if use_mtime:
			ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
		else:
			regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
			if regex_match and regex_match.groups():
				ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

	checkpoints_sorted = sorted(ordering_and_checkpoint_path)
	checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
	return checkpoints_sorted


'''
    Earlustop class
'''
class EarlyStopper(object):

    def __init__(self, num_trials, save_path, record):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.record = record

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if self.record == 1:
                torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)