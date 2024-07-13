# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from .phi2.modeling_phi import PhiConfig, PhiModel, PhiForCausalLM,PhiPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class ImpConfig(PhiConfig):
    model_type = "imp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_token_index = getattr(self, "image_token_index", 50296)
        self.image_token = getattr(self, "image_token", "<image>")


class ImpModel(LlavaMetaModel, PhiModel):
    config_class = ImpConfig

    def __init__(self, config: ImpConfig):
        super(ImpModel, self).__init__(config)


class ImpForCausalLM(PhiPreTrainedModel, LlavaMetaForCausalLM):
    """Imp for Causal Language Modeling."""

    # _keys_to_ignore_on_load_missing = [""]
    # _keys_to_ignore_on_load_unexpected = [r"transformer\.h\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]
    config_class = ImpConfig

    def __init__(self, config: ImpConfig) -> None:
        super().__init__(config)

        self.model = ImpModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def get_model(self):
        return self.model

    def get_decoder(self):
        return self.model
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def image_preprocess(self, images):
        return self.get_vision_tower().image_processor(images)['pixel_values']

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
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                images,
                'phi2'
            )

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values, 
            attention_mask=attention_mask,
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            loss = None
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
