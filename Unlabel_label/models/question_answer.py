import torch.nn as nn
import torch
from typing import List, Optional
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss


class ExtenedQuestionAnswering(nn.Module):
    def __init__(self, config):
        self.num_labels = config.num_labels
        print(self.num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        
    def forward(self, 
                base_model,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                start_positions: Optional[torch.LongTensor] = None,
                end_positions: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        outputs = base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return (total_loss, start_logits, end_logits)
    

def run_qa_batch(base_model, model, batch, args, try_again=True):
    try :
        # print(tokenizer.decode(list(batch[0])))
        inputs, segment_ids, input_mask, labels = batch	
        # print(tokenizer.decode(list(inputs[0])))
        # print(tokenizer.decode([1 if i<0 else i fo r i in list(labels[0])]))
        inputs = inputs.to(args.device)
        segment_ids = segment_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        labels = labels.to(args.device)
        model.train()
        base_model.train()
        outputs = model(base_model.roberta, inputs, token_type_ids=segment_ids, attention_mask=input_mask, labels=labels)
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
            outputs = run_qa_batch(base_model, model, batch, args, try_again=False)
    return outputs
