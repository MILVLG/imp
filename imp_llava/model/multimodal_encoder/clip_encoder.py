# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.

# Adopted from https://github.com/haotian-liu/LLaVA.

import torch
import torch.nn as nn

from typing import Dict, Optional, Union
import numpy as np

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from .siglip.image_processing_imp import ImpImageProcessor
from .siglip.modeling_siglip import SiglipVisionModel
from .siglip.configuration_siglip import SiglipVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            return
        
        # It's a hacky way to check if model is initialized under meta device
        # context, which will be enabled when loading trained model by huggingface 
        # `from_pretrained` api. In the case that a full model with vision tower is
        # loaded, there will be a warning if vision tower is loaded to cpu here. So we
        # set `device_map` to `auto` in order to avoid the warning.
        # [Edited by zhenwei - 2024-02-02 13:03]
        is_meta = getattr(nn.Linear(1, 1, bias=False).weight, 'is_meta', False)
        if 'siglip' in self.vision_tower_name:
            # "google/siglip-so400m-patch14-384"
            self.image_processor = ImpImageProcessor()
            if is_meta:
                # cfg = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
                # self.vision_tower = SiglipVisionModel(cfg)
                self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map='auto')
            else:
                self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
            del self.vision_tower.vision_model.encoder.layers[(self.select_layer + 1):]
            self.vision_tower.vision_model.post_layernorm = nn.Identity()
            self.vision_tower.vision_model.head = nn.Identity()
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            if is_meta:
                # cfg = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
                # self.vision_tower = CLIPVisionModel(cfg)
                self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map='auto')
            else:
                self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            del self.vision_tower.vision_model.encoder.layers[(self.select_layer + 1):]
        
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_forward_outs.hidden_states[-1]
        if self.select_feature == 'patch':
            image_features = image_features[:, -self.num_patches:]
            assert image_features.shape[-2] == self.num_patches, f'select_feature=patch, image_features.shape[-2]={image_features.shape[-2]} != num_patches={self.num_patches}'
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            assert image_features.shape[-2] == self.num_patches + 1, f'select_feature=cls_patch, image_features.shape[-2]={image_features.shape[-2]} != num_patches+1={self.num_patches+1}'
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # assert self.num_patches == 729
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                # image_feature = image_forward_out.last_hidden_state.to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # image_features = image_forward_outs.last_hidden_state.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device
    
    @property
    def is_meta(self):
        return self.device.type == 'meta'

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
