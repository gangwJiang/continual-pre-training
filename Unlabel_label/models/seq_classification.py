import torch.nn as nn
import torch
from typing import List, Optional
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ExtenedSeqClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        print("num_labels", self.num_labels)
        self.classifier = ClassificationHead(config)
        self.problem_type = None
        
    def forward(self, 
                base_model,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None):
        
        outputs = base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return (loss, logits)
    

def run_seq_cla_batch(base_model, model, batch, args, try_again=True):
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
            outputs = run_seq_cla_batch(base_model, model, batch, args, try_again=False)
    return outputs
