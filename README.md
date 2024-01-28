# 😈 imp

> A very small man can cast a very large shadow.
> 
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;——*George R.R. Martin, A Clash of Kings*


\[Technical report (coming soon)\]&nbsp;&nbsp;[[Demo](https://xmbot,net/imp/)\]&nbsp;&nbsp;[[Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/)\]


The Imp project aims to provide a family of  a strong multimodal `small` language models (MSLMs). Our `imp-v1-3b` is a strong MSLM with only **3B** parameters, which is build upon a small yet powerful SLM [Phi-2](https://huggingface.co/microsoft/phi-2) (2.7B) and a powerful visual encoder [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) (0.4B), and trained on the [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) training set.  

As shown in the Table below, `imp-v1-3b` significantly outperforms the counterparts of similar model sizes, and even achieves slightly better performance than the strong LLaVA-7B model on various multimodal benchmarks. 

We release the model weights a running example of `imp-v1-3b` on [Huggingface](https://huggingface.co/MILVLG/imp-v1-3b/). Technical report and training and evaluation code will be released soon. We will persistently improve our model and release the next versions to further improve model performance :)

## Model evaluation
We conduct evaluation on 9 commonly-used benchmarks, including 5 academic VQA benchmarks and 4 popular MLLM benchmarks, to compare our Imp model with LLaVA (7B) and existing MSLMs of similar model sizes.

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
  author = {Shao, Zhenwei and Ouyang, Xuecheng and Yu, Zhou and Yu, Jun},
  title = {Imp-v1: An emprical study of multimodal small language models},
  year = {2024},
  url = {https://huggingface.co/MILVLG/imp-v1-3b}
}
```
