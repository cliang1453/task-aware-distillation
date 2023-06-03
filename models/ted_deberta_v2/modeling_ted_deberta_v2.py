# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch TEDDeBERTa-v2 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
)

from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2ForQuestionAnswering,
    DebertaV2Model,
    DebertaV2Encoder,
    DebertaV2Layer,
)

@dataclass
class TEDBaseModelOutput(BaseModelOutput):
    filter_states: Tuple[torch.FloatTensor] = None

@dataclass
class TEDQuestionAnsweringModelOutput(QuestionAnsweringModelOutput):
    filter_states: Tuple[torch.FloatTensor] = None

class TEDDebertaV2Layer(DebertaV2Layer):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        
        assert layer_idx is not None
        self.should_add_filter = (
            (layer_idx < config.num_hidden_layers//2 and layer_idx % config.filter_interval == 0) 
            or (layer_idx >= config.num_hidden_layers//2 and layer_idx % config.filter_interval == config.filter_interval - 1)
        )
        
        if self.should_add_filter:
            filter_output_dim = config.filter_output_dim if config.filter_output_dim else config.hidden_size
          
            if config.filter_nonlinear:
                self.filter = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size), 
                    ACT2FN[config.hidden_act],
                    nn.Linear(config.hidden_size, filter_output_dim),
                )
            else:
                self.filter = nn.Linear(config.hidden_size, filter_output_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        filter_layer_output = None
        if self.should_add_filter:
            filter_layer_output = self.filter(layer_output)

        if output_attentions:
            return ((layer_output, filter_layer_output), att_matrix)
        else:
            return (layer_output, filter_layer_output)

class TEDDebertaV2Encoder(DebertaV2Encoder):
    
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([TEDDebertaV2Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv

        all_filter_states = ()
        filter_states = None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            
            if filter_states is not None:
                all_filter_states = all_filter_states + (filter_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                output_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states
            
            output_states, filter_states = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if filter_states is not None:
            all_filter_states = all_filter_states + (filter_states,)

        if not return_dict:
            return tuple(v for v in [
                output_states, all_hidden_states, all_attentions] if v is not None)
        return TEDBaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions, 
            filter_states=all_filter_states,
        )

# Copied from transformers.models.deberta.modeling_deberta.DebertaV2Model with BaseModelOutput->TEDBaseModelOutput
class TEDDebertaV2Model(DebertaV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = TEDDebertaV2Encoder(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        return TEDBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
            filter_states=encoder_outputs.filter_states,
        )

class TEDDebertaV2ForQuestionAnswering(DebertaV2ForQuestionAnswering):

    def __init__(self, config):
        super().__init__(config)
        self.deberta = TEDDebertaV2Model(config)
        if config.train_filters:
            self.num_filters = config.num_hidden_layers // config.filter_interval
            filter_output_dim = config.filter_output_dim if config.filter_output_dim else config.hidden_size
            self.filter_head = nn.ModuleList([
                nn.Linear(filter_output_dim, config.num_labels) for _ in range(self.num_filters)
            ])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not self.config.train_filters:
            sequence_output = outputs[0]
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
        else:
            start_logits, end_logits = None, None
            filter_start_end_logits = []
            for filter_idx in range(self.num_filters):
                filter_logits = self.filter_head[filter_idx](outputs['filter_states'][filter_idx])
                filter_start_logits, filter_end_logits = filter_logits.split(1, dim=-1)
                filter_start_logits = filter_start_logits.squeeze(-1).contiguous()
                filter_end_logits = filter_end_logits.squeeze(-1).contiguous()
                filter_start_end_logits.append((filter_start_logits, filter_end_logits))

        total_loss = None
        if start_positions is not None and end_positions is not None: 
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            if not self.config.train_filters:
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
            else:
                total_loss = 0.0
                for filter_start_logits, filter_end_logits in filter_start_end_logits:
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = filter_start_logits.size(1)
                    _start_positions = start_positions.clamp(0, ignored_index)
                    _end_positions = end_positions.clamp(0, ignored_index)
                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(filter_start_logits, _start_positions)
                    end_loss = loss_fct(filter_end_logits, _end_positions)
                    total_loss += (start_loss + end_loss) / 2
                total_loss /= self.num_filters

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TEDQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            filter_states=outputs.filter_states,
        )