# 😈 imp

> A very small man can cast a very large shadow.
> 
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;——*George R.R. Martin, A Clash of Kings*


\[Technical report (coming soon)\]&nbsp;&nbsp;[[Demo](https://xmbot.net/imp/)\]&nbsp;&nbsp;[[Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/)\]


This repository contains the official training/evaluation code of the Imp project, which aims to provide a family of  a strong multimodal small language models (MSLMs). Our `imp-v1-3b` is a strong MSLM with only **3B** parameters, which is build upon a small yet powerful SLM [Phi-2](https://huggingface.co/microsoft/phi-2) (2.7B) and a powerful visual encoder [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) (0.4B), and trained on the [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) training set.  

As shown in the [Evaluation](#evaluation), `imp-v1-3b` significantly outperforms the counterparts of similar model sizes, and even achieves slightly better performance than the strong LLaVA-7B model on various multimodal benchmarks. 

We also release the model weights a running example of `imp-v1-3b` on [Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/). Technical report will be released soon. We will persistently improve our model and release the next versions to further improve model performance :)

## Updates
- February 7, 2024: Training and evaluation codes of `imp-v1-3b` are released.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)
- [Citation](#citation)
<!-- - [Acknowledgement](#acknowledgement) -->

## Prerequisites

1. Clone this repository and navigate to the folder 
``` shell
git clone https://github.com/MILVLG/imp.git
cd imp
```
2. Install Package
We recommend using [Anaconda](https://www.anaconda.com/) to create a new environment for the project, and install the requirements with the following commands:
``` shell
conda create -n imp python=3.10 -y
conda activate imp
pip install -r requirements.txt
pip install flash-attn==2.4.2 --no-build-isolation
```
3. (Optional) Manually download the pretrained model repositories, i.e., [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) and [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) to your local directories and modify the corresponding paths in training and evaluation scripts.

## Training
The training pipeline and datasets of `imp-v1-3b` are directly inherited from [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). The training  
- *Multimodal pretraining*: train a projector on a subset of ∼558K image-text pairs to connect a frozen pretrained vision encoder and a frozen LLM.
- *Multimodal instruction tuning*: fine-tune the projector and LoRA in the LLM with multimodal instruction data and VQA-formatted data to empower the MLSM the ability of multimodal instruction following.

Imp is trained on 8 A100 (40G) GPUs. You can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` to match your resources. .But always keep the global batch size the same: `global_batch_size ` = `per_device_train_batch_size` $`\times`$ `gradient_accumulation_steps` $`\times`$ `num_gpus`.

<details>
<summary>Training scripts </summary>

### Stage-1: Multimodal pretraining

Please download the caption annotations `blip_laion_cc_sbu_558k.json` and images from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). Move the downloaded files to the `./datasets` folder, with image folder unzipped and renamed to `pretrain_images`. Then run the following command to start the training process:

``` shell
bash scripts/pretrain.sh
```

After that, a checkpoint of the multimodal projector will be stored in `./checkpoints/imp-v1-3b-pretrain`.

### Stage-2: Multimodal instruction tuning

Please download the annotation file of the mixed instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows:

```
datasets
├── llava_v1_5_mix665k.json
└── finetune_images
    ├── coco
    │   └── train2017
    ├── gqa
    │   └── images
    ├── ocr_vqa
    │   └── images
    ├── textvqa
    │   └── train_images
    └── vg
        ├── VG_100K
        └── VG_100K_2
```

Then start the training process with the following command:

``` shell
bash scripts/finetune_lora.sh
# bash scripts/finetune.sh # fully finetuning is not recommended
```
You will get a trained model (a LoRA diff if you use `finetune_lora.sh`) under `./checkpoints/` when the training is done.
</details>


## Evaluation
We follow the evaluation of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main) and conduct experiments on 9 commonly-used benchmarks, including 5 academic VQA benchmarks and 4 popular MLLM benchmarks. All evaluation scripts are placed in the `scripts/eval` folder. 

Before preparing task-specific data, you should download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and extract to `./playground/data/eval`. For more specific instructions, please refer to [LLaVA's Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 

It is supported to evaluate your own trained checkpoints or our official released `imp-v1-3b` at [Huggingface Hub](https://huggingface.co/MILVLG/imp-v1-3b/). 

See [Evaluation.md](./docs/Evaluation.md).

Using the script in `Evaluation.md`, we obtain our Imp model's accuracies in the following table, Compare with LLaVA (7B) and existing MSLMs of similar model sizes.

| Models | VQAv2 | GQA |VizWiz  | SQA(IMG) | TextVQA | POPE |  MME(P) | MMB  |MM-Vet|
|:--------:|:----:|:----:|:-------------:|:--------:|:-----:|:----:|:-------:|:-------:|:-------:|
| [LLaVA-v1.5-lora](https://github.com/haotian-liu/LLaVA) (7B) |79.10 | **63.00** |47.80 |  68.40 |58.20| 86.40 | **1476.9** | 66.10  |30.2|
| [TinyGPT-V](https://github.com/DLYuanGod/TinyGPT-V) (3B) | - | 33.60  | 24.80  |    -   |    -  | -| - | -  |-|
| [LLaVA-Phi](https://github.com/zhuyiche/llava-phi) (3B) | 71.40  | - | 35.90 |    68.40   |    48.60  | 85.00 | 1335.1 | 59.80 |28.9|
| [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM) (3B) | - | 59.00  | - |    61.00   |    47.50   | 84.90 | 1288.9 | 59.60  |-|
| [MC-LLaVA](https://huggingface.co/visheratin/MC-LLaVA-3b) (3B) | 64.24 | 49.60  | 24.88 |    -   |    38.59   | 80.59 | - | -  |-|
| **Imp-v1 (3B, ours)** | **79.45**  | 58.55 | **50.09** |**69.96**| **59.38** | **88.02**| 1434.0 | **66.49**  |**33.1**|

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

## About us
This project is maintained by the [MILVLG](https://github.com/MILVLG)@Hangzhou Dianzi University (HDU) led by Prof. Zhou Yu and Jun Yu, and is mainly developed by Zhenwei Shao and Xuecheng Ouyang. We hope our model may serve as a strong baseline to inspire future research on MSLM, as well as its derivative applications on mobile devices and robots. 

## Citation

If you use our model or refer our work in your studies, please cite:

```bibtex
@misc{imp2024,
  author = {Shao, Zhenwei and Ouyang, Xuecheng and Gai, Zhenbiao and Yu, Zhou and Yu, Jun},
  title = {Imp: An emprical study of multimodal small language models},
  year = {2024},
  url = {https://huggingface.co/MILVLG/imp-v1-3b}
}
```
