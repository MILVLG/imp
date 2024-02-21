# Model Zoo
We provide the checkpoints of different Imp models in this page.

### Imp-v1-3b

- If you just want to run model inference or evaluation, you can download the integrated single model model on [Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/).
- If you want to run custom model fine-tuning, you can alternatively download checkpoints of our separated sub-models as follows: 
  - Stage-1: [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETkchWF9vrJLoGz8gd3Klx4BDCZnBkESpDz1f9DegGC_7g?download=1)
  - Stage-2 (LoRA): [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EcpJj4ZBvlBNuHL_yXXRhwUBZl6QLuCQWHQ8KUrh_ui1iA?download=1)

After downloading all of them, the checkpoints can be organized as the structure as follows. Note that the stage-1 model is not used if you want to start fine-tuning from the stage-2 model. 

```
checkpoints
├── base
│   └── siglip-so400m-patch14-384
│   └── phi-2
├── imp-v1-3b-stage1
│   └── mm_projector.bin
└── imp-v1-3b-stage2-lora
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── configuration_phi.py
    ├── non_lora_trainables.bin
    └── trainer_state.json
```