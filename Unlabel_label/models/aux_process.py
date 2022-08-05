import torch
from transformers import PreTrainedTokenizer



'''
	Core MLM functionality
'''
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args):
	""" Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

	if tokenizer.mask_token is None:
		raise ValueError(
			"This tokenizer does not have a mask token which is necessary for masked"
			" language modeling. Remove the --mlm flag if you want to use this tokenizer."
		)

	labels = inputs.clone()
	# We sample a few tokens in each sequence for masked-LM training
	# (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
	probability_matrix = torch.full(labels.shape, args.mlm_probability)
	special_tokens_mask = [
		tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
	]
	probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
	if tokenizer._pad_token is not None:
		padding_mask = labels.eq(tokenizer.pad_token_id)
		probability_matrix.masked_fill_(padding_mask, value=0.0)
	masked_indices = torch.bernoulli(probability_matrix).bool()
	labels[~masked_indices] = -100  # We only compute loss on masked tokens

	# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
	indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
	inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

	# 10% of the time, we replace masked input tokens with random word
	indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
	random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
	inputs[indices_random] = random_words[indices_random]

	# The rest of the time (10% of the time) we keep the masked input tokens unchanged
	return inputs, labels


# Run a batch of data through the model whilst checking for out of memory errors
# Process a batch of data for a particular auxiliary task
def run_aux_batch(model, batch, tokenizer, args, task_name, try_again=True):
	try :
		# print(tokenizer.decode(list(batch[0])))
		inputs, labels = mask_tokens(batch, tokenizer, args) if task_name=="mlm" else (batch, batch)	
		# print(tokenizer.decode(list(inputs[0])))
		# print(tokenizer.decode([1 if i<0 else i fo r i in list(labels[0])]))
		inputs = inputs.to(args.device)
		labels = labels.to(args.device)
		model.train()
		# print(model)
		outputs = model(inputs, labels=labels)
	except RuntimeError as e:
		if 'out of memory' in str(e):
			if try_again:
				print('| WARNING: ran out of memory during forward. Trying batch again')
			else:
				print('| WARNING: ran out of memory during forward. Skipping batch')
		else:
			print('Run into this new error : ', str(e))
		torch.cuda.empty_cache()
		if not try_again:
			return None
		else:
			outputs = run_aux_batch(model, batch, tokenizer, args, task_name, try_again=False)
	return outputs

	
	# return loss_