# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.

# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from imp_llava.model import *
from imp_llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from imp_llava.model.multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from imp_llava import logger

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    logger.info(f'load cfg kwargs: {kwargs}')
    if 'llava' in model_name.lower() or 'imp' in model_name.lower():
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            exit()
        if 'lora' in model_name.lower() and model_base is not None:
            # Load model trained with LoRA
            logger.info(f'Load model name trained with LoRA, model base: {model_base}')
            assert 'imp' in model_name.lower(), 'The model name must contain `imp`'
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
            if 'phi-2' in model_name.lower() or  'phi2' in model_name.lower():
                    lora_cfg_pretrained = ImpConfig.from_pretrained(model_path)
                    model = ImpForCausalLM.from_pretrained(model_base, config=lora_cfg_pretrained, **kwargs)
            elif 'qwen1.5' in model_name.lower():
                lora_cfg_pretrained = ImpQwen2Config.from_pretrained(model_path, trust_remote_code=True)
                model = ImpQwen2ForCausalLM.from_pretrained(model_base, config=lora_cfg_pretrained, **kwargs)
            elif'phi3' in model_name.lower():
                lora_cfg_pretrained = ImpPhi3Config.from_pretrained(model_path, trust_remote_code=True)
                model = ImpPhi3ForCausalLM.from_pretrained(model_base, config=lora_cfg_pretrained, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            logger.info('Loading additional weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith(f'model.{model.base_model_prefix}.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            logger.info(f'Loading additional weights: f{[*non_lora_trainables.keys()]}')
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            logger.info('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            logger.info('Merging LoRA weights...')
            model = model.merge_and_unload()
            logger.info('Model is loaded...')
        elif model_base is not None:
            logger.info('Load mm projector only model...')
            if 'phi2' in model_name.lower() or 'phi-2' in model_name.lower():
                logger.info(f'model_base:, {model_base}')
                config = ImpConfig.from_pretrained(model_path, trust_remote_code=True)
                model = ImpForCausalLM.from_pretrained(model_base, **kwargs)
                model.model.vision_tower = build_vision_tower(config)
                model.model.mm_projector = build_vision_projector(config)
                tokenizer = AutoTokenizer.from_pretrained(model_base)
            elif 'qwen1.5' in model_name.lower():
                logger.info(f'model_base:, {model_base}')
                config = ImpQwen2Config.from_pretrained(model_path, trust_remote_code=True)
                model = ImpQwen2ForCausalLM.from_pretrained(model_base, **kwargs)
                model.model.vision_tower = build_vision_tower(config)
                model.model.mm_projector = build_vision_projector(config)
                tokenizer = AutoTokenizer.from_pretrained(model_base)
            elif 'phi3' in model_name.lower():
                logger.info(f'model_base:, {model_base}')
                config = ImpPhi3Config.from_pretrained(model_path, trust_remote_code=True)
                model = ImpPhi3ForCausalLM.from_pretrained(model_base, **kwargs)
                model.model.vision_tower = build_vision_tower(config)
                model.model.mm_projector = build_vision_projector(config)
                tokenizer = AutoTokenizer.from_pretrained(model_base)
            
            else:
                logger.info(f'model_base:, {model_base}')
                config = ImpConfig.from_pretrained(model_path, trust_remote_code=True)
                model = ImpForCausalLM.from_pretrained(model_base, **kwargs)
                model.model.vision_tower = build_vision_tower(config)
                model.model.mm_projector = build_vision_projector(config)
                tokenizer = AutoTokenizer.from_pretrained(model_base)
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            logger.info(f'loading mm projector weights: {[*mm_projector_weights.keys()]}')
            model.load_state_dict(mm_projector_weights, strict=False)
            # model.to(device)
            logger.info('Model is loaded...')
        else:
            logger.info(f'load fully fine-tuned model or HF Hub model: {model_path}')
            #hg version
            
            if 'phi2' in model_name.lower() or 'phi-2' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = ImpForCausalLM.from_pretrained(model_path, **kwargs)
                logger.info('Model is loaded...')
            elif 'qwen1.5' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = ImpQwen2ForCausalLM.from_pretrained(model_path, **kwargs)
                logger.info('Model is loaded...')
            elif 'phi3' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = ImpPhi3ForCausalLM.from_pretrained(model_path, **kwargs)
                logger.info('Model is loaded...')
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = ImpForCausalLM.from_pretrained(model_path, **kwargs)
                logger.info('Model is loaded...')
    else:
        raise NotImplementedError
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            logger.info(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            logger.info(f"Merging weights")
            model = model.merge_and_unload()
            logger.info('Convert to FP16...')
            model.to(torch.float16)
        else:
            if 'phi2' in model_name.lower() or 'imp' in model_name.lower():
                raise NotImplementedError
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)



    image_processor = None

    if 'llava' in model_name.lower()  or 'imp' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        # FIXME: phi-2 has unused embeddings.
        # [Edited by zhenwei - 2024-01-31 13:50]
        # model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            logger.info('Delayed vision tower loaded.')
        vision_tower.to(device=model.device, dtype=model.dtype)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
