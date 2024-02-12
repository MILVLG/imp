# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from .phi2.modeling_phi import PhiConfig, PhiModel, PhiForCausalLM, CausalLMHead, CausalLMLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class ImpConfig(PhiConfig):
    model_type = "imp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_function = getattr(self, 'hidden_act', 'gelu_new')
        self.attn_pdrop = getattr(self, 'attention_dropout', 0.0)
        self.layer_norm_epsilon = getattr(self, 'layer_norm_eps', 1e-5)
        self.n_embd = getattr(self, 'hidden_size', 2560)
        self.n_head = getattr(self, 'num_attention_heads', 32)
        self.n_layer = getattr(self, 'num_hidden_layers', 32)
        self.n_positions = getattr(self, 'max_position_embeddings', 2048)
        self.flash_attn = getattr(self, 'flash_attn', False)
        self.flash_rotary = getattr(self, 'flash_rotary', False)
        self.fused_dense = getattr(self, 'fused_dense', False)
        self.n_head_kv = getattr(self, 'num_key_value_heads', None)
        self.n_inner = getattr(self, 'intermediate_size', None)
        self.rotary_dim = 32
        # self.image_token_index = getattr(self, "image_token_index", 50296)
        # self.image_token = getattr(self, "image_token", "<image>")


class ImpModel(LlavaMetaModel, PhiModel):
    config_class = ImpConfig

    def __init__(self, config: ImpConfig):
        super(ImpModel, self).__init__(config)


class ImpForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = ImpConfig

    def __init__(self, config):
        super(ImpForCausalLM, self).__init__(config)
        self.transformer = ImpModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = CausalLMHead(config)
        self.loss = CausalLMLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("imp", ImpConfig)
AutoModelForCausalLM.register(ImpConfig, ImpForCausalLM)
