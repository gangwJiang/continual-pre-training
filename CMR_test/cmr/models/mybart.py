# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This script was based on https://github.com/shmsw25/bart-closed-book-qa.
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
# from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_bart import shift_tokens_right

from .utils import label_smoothed_nll_loss


class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
                use_cache=False, is_training=False, return_all_loss=False, past_key_values=None,
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, return_dict=True,
                output_attentions=None, output_hidden_states=None):

        if is_training:
            # print(decoder_input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1)
            _decoder_input_ids = shift_tokens_right(
                # decoder_input_ids, self.config.pad_token_id, decoder_input_ids.ne(self.config.pad_token_id).sum(dim=1) - 1)
                decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(
            outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, decoder_input_ids, epsilon=0.1, ignore_index=self.config.pad_token_id, return_all_loss=return_all_loss)
            return loss
        return (lm_logits, ) + outputs[1:]
