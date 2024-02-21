# Model Zoo
We provide the checkpoints of different Imp models in this page.

### Imp-v1-3b
- If you only want to run model inference or evaluation, you can download our merged model from [Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/).
- If you want to run model stage2 fine-tuning or custom fine-tuning on your own dataset, you should download the following checkpoints, respectively. 
  - Stage-1: [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETkchWF9vrJLoGz8gd3Klx4BDCZnBkESpDz1f9DegGC_7g?download=1)
  - Stage-2 (LoRA): [ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EcpJj4ZBvlBNuHL_yXXRhwUBZl6QLuCQWHQ8KUrh_ui1iA?download=1)

After downloading all of them, the `checkpoints` folder is organized as follows. If you want to run custom fine-tuning, you need to merge the stage2 sub-models into a single model. 

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
