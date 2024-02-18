# Model Card
We provide the checkpoints of different Imp models in this page.

### Imp-v1-3b

- If you just want to run model inference or evaluation, you can download the integrated single model model on [Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/).
- If you want to run custom model fine-tuning, you can alternatively download checkpoints of our separated sub-models as follows: 
  - Base models: 
    - Visual Encoder: [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
    - LLM: [Phi-2](https://huggingface.co/microsoft/phi-2)
  - Stage-1: [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EfEI93KFam5KplL1AWKgldUBjQVotT4kxQS63UlROODouA?download=1)
  - Stage-2: [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EcpJj4ZBvlBNuHL_yXXRhwUBZl6QLuCQWHQ8KUrh_ui1iA?download=1)

After downloading all of them, the checkpoints can be organized as the structure as follows. Note that the stage-1 model is not used if you want to start fine-tuning from the stage-2 model. 

```
checkpoints
├── base
│   └── siglip-so400m-patch14-384
│   └── phi-2
├── imp-v1-3b-pretrain
│   └── mm_projector.bin
└── imp-v1-3b-lora
    ├── adapter_config.json
    ├── adapter_model.bin
    ├── config.json
    ├── configuration_phi.py
    ├── non_lora_trainables.bin
    └── trainer_state.json
```