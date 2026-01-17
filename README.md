# Post-Hoc Control over Mixture-of-Experts

This repository implements the main experiments of our TACL 2024 paper, [Not Eliminate but Aggregate: Post-Hoc Control over Mixture-of-Experts to Address Shortcut Shifts in Natural Language Understanding](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00701/124836/Not-Eliminate-but-Aggregate-Post-Hoc-Control-over).

The code is intended solely for reproducing the experiments. We thank the authors of [RISK](https://github.com/CuteyThyme/RISK), on which our code was based.


## Environment

We tested our code in the following environment.
* OS: Debian GNU/Linux 10 (buster)
* Python: 3.8.3
* CUDA: 11.2
* GPUs: **NVIDIA V100 x 2**

The experiment with `DeBERTa-v3-large` requires a different environment.
* OS: Debian GNU/Linux 10 (buster)
* Python: 3.8.3
* CUDA: 11.2
* GPUs: **NVIDIA A100 (40GB) x 2**


## Getting Started

```bash
git clone https://github.com/CyberAgentAILab/posthoc-control-moe
cd posthoc-control-moe
```


### Installation

> [!NOTE]
> The exact versions of the libraries we used are specified in the requirements for reproducibility. For improved security, consider updating the libraries, particularly PyTorch and Transformers. However, note that we have not tested reproducibility with the updated versions.  

Install dependencies to reproduce the main results.  
```bash
# For conda users
conda env create -f environment.yaml
conda activate posthoc-control-moe

# For the others
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

For the experiment with `DeBERTa-v3-large`, use `environment_deberta.yaml` or `requirements_deberta.txt`.
```bash
# For conda users
conda env create -f environment_deberta.yaml
conda activate posthoc-control-moe-deberta

# For the others
pip install --force-reinstall --no-cache-dir -r requirements_deberta.txt
```


### Data Preparation

[Download the datasets from here](https://drive.google.com/drive/folders/1aleJytl3SAKdGBsxZbxznwusINOnTAzh?usp=share_link) and place them as follows.
Or you can just run `gdown 'https://drive.google.com/drive/folders/1aleJytl3SAKdGBsxZbxznwusINOnTAzh?usp=share_link' --folder` to download the datasets at once.
The link is kindly provided by [RISK](https://github.com/CuteyThyme/RISK).
```
./dataset/
  ├── multinli/
  │     ├── train.tsv
  │     └── dev_matched.tsv
  ├── hans/heuristics_evaluation_set.txt
  ├── qqp_paws/
  │     ├── qqp_train.tsv
  │     ├── qqp_dev.tsv
  │     └── paws_devtest.tsv
  └── fever/
        ├── fever.train.jsonl
        ├── fever.dev.jsonl
        ├── symmetric_v0.1/fever_symmetric_generated.jsonl
        └── symmetric_v0.2/fever_symmetric_test.jsonl
```

Original links for the datasets:
* MNLI:  [https://cims.nyu.edu/~sbowman/multinli/](https://cims.nyu.edu/~sbowman/multinli/)     
* HANS:  [https://github.com/tommccoy1/hans](https://github.com/tommccoy1/hans)    
* QQP and PAWS: [https://github.com/google-research-datasets/paws](https://github.com/google-research-datasets/paws)
* FEVER and FEVER-Symmetric: [https://github.com/TalSchuster/FeverSymmetric](https://github.com/TalSchuster/FeverSymmetric)     


## Usage

### Training

Train the mixture-of-experts and save the one that performs the best on ID dev.
Here, we specify the seed that yields near the average performance shown in the paper.
The default seed is `777`, and the analyses were conducted on that seed.
```bash
mkdir -p saved_models/mnli
mkdir -p saved_models/qqp
mkdir -p saved_models/fever

# MNLI
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset mnli --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 0.5 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/mnli \
    --best_model_name bert_mos_e10_rs05k8_ep10_lr2e-5_8 --save

# QQP
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset qqp --batch_size 32 --epochs 10 \
    --num_experts 15 --router_loss 1 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/qqp \
    --best_model_name bert_mos_e15_rs1k8_ep10_lr2e-5_8 --save

# FEVER
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset fever --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 1 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/fever \
    --best_model_name bert_mos_e10_rs1k8_ep10_lr2e-5_8 --save
```

For the `DeBERTa-v3-large` ablation study:
```bash
# Make sure to use the environment and dependencies prepared for DeBERTa-v3-large
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_deberta.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path microsoft/deberta-v3-large \
    --dataset mnli --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 0.5 --router_tau 1 \
    --num_topk_mask 8 --lr 5e-6 --max_grad_norm 1 --seed 888 \
    --save_dir saved_models/mnli \
    --best_model_name deberta_mos_e10_rs05k8_ep10_lr5e-6g1_bf16_8 --save
```


### Evaluation

Evaluate the post-hoc control over the mixture-of-experts on OOD tests.
[Some saved models are available here](https://console.cloud.google.com/storage/browser/ailab-public/posthoc-control-moe) for those who want to check the results quickly.
Download and place them under `saved_models/[task_name]/`.

```bash
# HANS
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset mnli --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 0.5 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/mnli \
    --resume bert_mos_e10_rs05k8_ep10_lr2e-5_8 --evaluate

# PAWS
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset qqp --batch_size 32 --epochs 10 \
    --num_experts 15 --router_loss 1 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/qqp \
    --resume bert_mos_e15_rs1k8_ep10_lr2e-5_8 --evaluate

# Symm. v1 and v2
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path bert-base-uncased \
    --dataset fever --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 1 --router_tau 1 \
    --num_topk_mask 8 --lr 2e-5 --seed 888 --save_dir saved_models/fever \
    --resume bert_mos_e10_rs1k8_ep10_lr2e-5_8 --evaluate
```

For the `DeBERTa-v3-large` ablation study:
```bash
# Make sure to use the environment and dependencies prepared for DeBERTa-v3-large
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file accelerate_config_deberta.yaml --main_process_port 20880 \
    src/main_mix.py --model bert_mos --pretrained_path microsoft/deberta-v3-large \
    --dataset mnli --batch_size 32 --epochs 10 \
    --num_experts 10 --router_loss 0.5 --router_tau 1 \
    --num_topk_mask 8 --lr 5e-6 --max_grad_norm 1 --seed 888 \
    --save_dir saved_models/mnli \
    --resume deberta_mos_e10_rs05k8_ep10_lr5e-6g1_bf16_8 --evaluate
```


## Citation

If you find our work useful for your research, please consider citing our paper:
```bibtex
@article{10.1162/tacl_a_00701,
    author = {Honda, Ukyo and Oka, Tatsushi and Zhang, Peinan and Mita, Masato},
    title = {Not Eliminate but Aggregate: Post-Hoc Control over Mixture-of-Experts to Address Shortcut Shifts in Natural Language Understanding},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {1268-1289},
    year = {2024},
    month = {10},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00701},
    url = {https://doi.org/10.1162/tacl\_a\_00701},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00701/2480600/tacl\_a\_00701.pdf},
}
```
