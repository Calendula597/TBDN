# TBDN

Official repository for **"Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding"**.

## Environment Setup

### Install PyTorch

```bash
conda create -n tbdn python=3.10.12
conda activate tbdn
```

### Install Dependencies

```bash
pip install -r ./cobsat/conda_env/default_requirements.txt
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Install FlashAttention and Transformers

```bash
pip install flash-attn --no-build-isolation
cd ./transformers
pip install .
```

### Dataset Download

Download the images and their corresponding descriptions from the **CoBSAT** dataset:

```bash
wget "https://huggingface.co/datasets/yzeng58/CoBSAT/resolve/main/datasets.zip"
```

Unzip `datasets.zip`, then move the extracted `datasets` folder into the `cobsat` directory.

## Usage 

### Inference
#### Option A: Limited compute

If your compute resources are limited (e.g., **2× NVIDIA RTX 4090**), please run **LVLM inference** first to generate textual descriptions, then run **Flux inference** to generate images:

```bash
python lvlm_inference.py
python flux_inference.py
```

#### Option B: Option B: Sufficient compute
If you have sufficient compute resources (e.g., **1× NVIDIA A100**), you can run the end-to-end pipeline directly:

```bash
python lvlm_flux_inference.py
```

### Evaluation

Before running the evaluation, please follow the CoBSAT documentation to configure the evaluation environment.

```bash
conda create -n llava python=3.10.13
conda activate llava

pip install --upgrade pip  # enable PEP 660 support
pip install git+https://github.com/yzeng58/LLaVA/@a61aae093656922fe16ec2152b031dd1de72fe92
pip install -r conda_env/llava_requirements.txt

python evaluation_icl.py \
  --model seed \
  --prompt_type default \
  --eval_mode image \
  --task_id 1 2 3 \
  --shot 2 4 \
  --device cuda \
  --seed 123 \
  --wandb 1 \
  --overwrite 0 \
  --finetuned_model 0 \
  --data_mode default \
  --manual_path /path/to/image
```

## Acknowledgements

Our codebase is built upon and/or adapted from the following excellent open-source projects:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [ThinkDiff](https://github.com/MiZhenxing/ThinkDiff)
- [VL-ICL](https://github.com/ys-zong/VL-ICL)
- [ICD](https://github.com/PostMindLab/ICD)
- [CoBSAT](https://github.com/UW-Madison-Lee-Lab/CoBSAT)

We sincerely thank the authors and contributors for making their code publicly available.


## Citation

If you find this project useful, please cite our paper:

```bibtex
@misc{ma2026thinkbrightdiffusenice,
  title         = {Think Bright, Diffuse Nice: Enhancing T2I-ICL via Inductive-Bias Hint Instruction and Query Contrastive Decoding},
  author        = {Zhiyong Ma and Zhenpeng Li and Yuanjie Shi and Zhengping Li and Jiahao Chen and Qingyuan Chuai},
  year          = {2026},
  eprint        = {2601.06169},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2601.06169}
}
```
