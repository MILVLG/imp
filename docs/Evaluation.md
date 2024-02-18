# Evaluation
We follow the evaluation of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main) and conduct experiments on 9 commonly-used benchmarks, including 5 academic VQA benchmarks and 4 popular MLLM benchmarks. All evaluation scripts are placed in the `scripts/eval` folder. 

Before preparing task-specific data, you should download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and extract to `./playground/data/eval`. For more specific instructions, please refer to [LLaVA's Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). 

You can choose to use `lora eval` or `merge eval` in evaluation scripts.
## Scripts
### VQAv2
1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash scripts/v1_5/eval/vqav2.sh
```
3. The result file could be found in `./playground/data/eval/vqav2/answers_upload`. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `./playground/data/eval/vizwiz`.
2. Inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/vizwiz.sh
```
3. The result file could be found in `./playground/data/eval/vizwiz/answers_upload`. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission)

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
```
3. The result file could be found in `./playground/data/eval/mmbench/answers_upload`. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal)

### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `./playground/data/eval/mmvet`.
2. Inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
```
3. Evaluate the predictions in `./playground/data/eval/mmvet/results` using the official jupyter notebook.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/gqa.sh
```

### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`.
2. Inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/pope.sh
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```