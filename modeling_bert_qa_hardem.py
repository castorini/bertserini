from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config, device, loss_type, variant_id=0, tau=None):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2) # [N, L, H] => [N, L, 2]
        #self.qa_classifier = nn.Linear(config.hidden_size, n_class) # [N, H] => [N, n_class]
        self.device = device
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)
        self.loss_type = loss_type
        self.tau = tau
        if self.loss_type=='hard-em':
            assert tau is not None

    def _forward(self, input_ids, attention_mask, token_type_ids):
        '''
        each batch is a list of 7 items (training) or 3 items (inference)
            - input_ids: token id of the input sequence
            - attention_mask: mask of the sequence (1 for present, 0 for blank)
            - token_type_ids: indicator of type of sequence.
            -      e.g. in QA, whether it is question or document
            - (training) start_positions: list of start positions of the span
            - (training) end_positions: list of end positions of the span
            - (training) switch: list of switches (can be used for general purposes.
            -      in this model, 0 means the answer is span, 1 means the answer is `yes`,
            -      2 means the answer is `no`, 3 means there's no answer
            - (training) answer_mask: list of answer mask.
            -      e.g. if the possible spans are `[0, 7], [3, 7]`, and your `max_n_answers` is 3,
            -      start_positions: [[0, 3, 0]]
            -      end_positions: [[7, 7, 0]]
            -      switch: [[0, 0, 0]]
            -      answer_mask: [[1, 1, 0]]
        '''
        all_encoder_layers = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #switch_logits = self.qa_classifier(torch.max(sequence_output, 1)[0])
        return start_logits, end_logits

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, answer_mask=None, position_ids=None, head_mask=None, global_step=-1):
        start_logits, end_logits = self._forward(input_ids, attention_mask, token_type_ids)
        if start_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            #answer_mask = answer_mask.type(torch.FloatTensor).to(self.device)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
            # You care about the span only when switch is 0
            #span_mask = answer_mask * (switch == 0).type(torch.FloatTensor).to(self.device)

            start_losses = [(loss_fct(start_logits, _start_positions) * _answer_mask) \
                            for (_start_positions, _answer_mask) \
                            in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]
            end_losses = [loss_fct(end_logits, _end_positions) * _answer_mask \
                            for (_end_positions, _answer_mask) \
                          in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
            assert len(start_losses) == len(end_losses)
            loss_tensor = \
                        torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                        torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

            if self.loss_type=='first-only':
                total_loss = torch.sum(start_losses[0]+end_losses[0]+switch_losses[0])
            elif self.loss_type == "hard-em":
                if numpy.random.random()<min(global_step/self.tau, 0.8):
                    total_loss = self._take_min(loss_tensor)
                else:
                    total_loss = self._take_mml(loss_tensor)
            elif self.loss_type == "mml":
                total_loss = self._take_mml(loss_tensor)
            else:
                raise NotImplementedError()
            return total_loss

        elif start_positions is None and end_positions is None:
            return start_logits, end_logits

        else:
            raise NotImplementedError()

    def _take_min(self, loss_tensor):
        return torch.sum(torch.min(
            loss_tensor + 2*torch.max(loss_tensor)*(loss_tensor==0).type(torch.FloatTensor).to(self.device), 1)[0])

    def _take_mml(self, loss_tensor):
        return torch.sum(torch.log(torch.sum(torch.exp(
                loss_tensor - 1e10 * (loss_tensor==0).float()), 1)))
