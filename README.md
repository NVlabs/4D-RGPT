# [CVPR 2026 (Highlight)] 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://www.ca-joe-yang.com/resource/projects/4D_RGPT/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.17012-b31b1b.svg)](https://arxiv.org/abs/2512.17012)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/nvidia/4D-RGPT-8B)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/nvidia/R4D-Bench)
[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue)](https://huggingface.co/papers/2512.17012)

This is the official repository for the **CVPR'26 Highlight** paper **[4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation](https://arxiv.org/abs/2512.17012)**.

[Chiao-An Yang*](https://www.ca-joe-yang.com/), [Ryo Hachiuma](https://scholar.google.com/citations?user=W1KPqGEAAAAJ), [Sifei Liu](https://sifeiliu.net/), [Subhashree Radhakrishnan](https://scholar.google.com/citations?user=bdlaQjkAAAAJ), [Raymond A. Yeh](https://raymond-yeh.com/), [Yu-Chiang Frank Wang](http://vllab.ee.ntu.edu.tw/ycwang.html), [Min-Hung Chen](https://minhungchen.netlify.app/) <br>
(*Work done during the internship at NVIDIA Research)

<h1 align="center">
    <img src="./teaser_4D-RGPT.png">
</h1>

---

## 📖 Introduction

Despite advances in Multimodal LLMs (MLLMs), their ability to reason over 3D structures and temporal dynamics remains limited, constrained by weak 4D perception and temporal understanding. Furthermore, existing 3D and 4D Video Question Answering (VQA) benchmarks emphasize static scenes and lack region-level prompting.

To tackle these issues, we introduce:
- **4D-RGPT**: A specialized MLLM designed to capture 4D representations from video inputs with enhanced temporal and spatial perception.
- **Perceptual 4D Distillation (P4D)**: A training-only framework that transfers 4D representations (e.g., depth, optical flow) from a frozen expert model into 4D-RGPT for comprehensive 4D perception—without introducing any additional inference cost.
- **R4D-Bench**: A rigorous benchmark for depth-aware dynamic scenes featuring region-level prompting, built via a hybrid automated and human-verified pipeline.

Our experiments demonstrate that 4D-RGPT achieves notable improvements over strong baselines on existing 3D/4D benchmarks (***+5.3%*** on average across 6 benchmarks) as well as our proposed region-based R4D-Bench (***+4.3%***).

## 💥 News 💥

- [x] **[Jun. 2026]** 🔥 Training + inference code and **[4D-RGPT-8B model weights](https://huggingface.co/nvidia/4D-RGPT-8B)** released.
- [x] **[Apr. 2026]** 🔥🔥 4D-RGPT is selected as a **Highlight** paper at **CVPR 2026**! 🎉🎉
- [x] **[Apr. 2026]** R4D-Bench Dataset release.
- [x] **[Feb. 2026]** 🔥🔥 4D-RGPT is accepted to **CVPR 2026**! 🎉🎉
- [x] **[Dec. 2025]** [Paper](https://arxiv.org/abs/2512.17012), [Project Page](https://www.ca-joe-yang.com/resource/projects/4D_RGPT/), and [Hugging Face page](https://huggingface.co/papers/2512.17012) released.

**Contact:** Chiao-An Yang — <yang2300@purdue.edu>

---

## Built on VILA

This repo is a fork of the official [NVlabs/VILA](https://github.com/NVlabs/VILA) and inherits the NVILA model code. The 4D-RGPT-specific additions are:

- **L_LD (latent distillation)** and **L_ED (explicit distillation)** from a frozen L4P teacher into NVILA's hidden states — see [`perception_distillation()` in `llava/model/language_model/llava_llama.py`](llava/model/language_model/llava_llama.py).
- **Timestamp Positional Encoding (TPE)** added to per-frame vision tokens — see [`llava/model/pe/time_pe.py`](llava/model/pe/time_pe.py) (wired in [`llava/model/llava_arch.py:107`](llava/model/llava_arch.py#L107)).
- **R4D-Bench** lmms-eval task with 12-column category breakdown — see [`tools/vlm/lmms-eval/ext/tasks/r4d_bench/`](tools/vlm/lmms-eval/ext/tasks/r4d_bench).

### Note on `transformers` version compatibility

Our code runs against **`transformers==4.46`** + `torch==2.3` (see [`pyproject.toml`](pyproject.toml)). We initially built on `transformers==5.0`, but pivoted back to 4.x: with `transformers==5.0`, SDPA on Qwen2-family LLMs produces a ~8pt bf16 argmax drift on close-call MCQ tokens (R4D-Bench: 30 vs the v4 reference 38.9 on NVILA-Lite-8B). The drift is in matmul kernel selection, not in the model weights.

Practical consequence: **a VILA checkpoint cannot be transferred between the v4 and v5 stacks as-is**, because transformers 5 only emits the fast `tokenizer.json`, while transformers 4.46's `AutoTokenizer(..., use_fast=False)` expects `vocab.json` / `merges.txt`. [`llava/model/language_model/builder.py:118-121`](llava/model/language_model/builder.py#L118-L121) handles this with a guarded slow→fast tokenizer fallback so most checkpoints load under either stack, but the saved artifacts still need to be re-emitted on the target stack.

The pinned `flash_attn==2.5.6+cu121torch2.3` wheel matters — later builds either fail `GLIBC_2.32 not found` or change SDPA fallbacks in ways that move accuracy.

---

## Quick start

1. **Setup**
   ```bash
   pip install uv && uv venv --python 3.11 && uv sync && source .venv/bin/activate
   cp .env.example .env        # fill in SLURM_ACCOUNT / DATASETS_ROOT
   ```
2. **Wire datasets** — edit `data/registry.yaml` (eval) and
   `llava/data/registry/datasets/<cluster>.yaml` (train). See
   [data/README.md](data/README.md).

### Train

```bash
# 1 GPU / debugging — sft.sh auto-detects single node and shrinks the batch.
bash scripts/nvila/sft.sh [BASE_MODEL] [RUN_NAME] [DATA_MIXTURE]
# defaults: BASE_MODEL=NVILA-Lite-8B, MIXTURE=sat+vstibench+robofac+wolf
```
SLURM (multi-node): `NUM_NODES=8 bash scripts/slurm/submit.sh scripts/nvila/sft.sh`.
See [scripts/nvila/sft.sh](scripts/nvila/sft.sh) for the full training-arg surface (TPE, L_LD/L_ED weights, region extractor, etc.).

### Eval

> **Released checkpoint:** [huggingface.co/nvidia/4D-RGPT-8B](https://huggingface.co/nvidia/4D-RGPT-8B) was trained on **`transformers==4.46`** (v4 stack). Load and run inference under the same stack — moving the checkpoint to `transformers==5.x` is not transparent (tokenizer artifacts differ; see [Note on `transformers` version compatibility](#note-on-transformers-version-compatibility) below). To extend or retrain, stay on the v4 stack as well.

```bash
# 4D-RGPT / NVILA-family
MODEL_PATH=runs/train/4D-RGPT-8B-v4 \
  bash scripts/lmms_eval/tasks/r4d_bench.sh

# Qwen2.5-VL baseline
bash scripts/lmms_eval/models/qwen2_5_vl.sh r4d_bench    # default model
```
Available tasks: `r4d_bench`, `vlm4d`, `vstibench` (under `scripts/lmms_eval/tasks/`).

R4D-Bench is the headline benchmark. Eval reports a 12-column accuracy table:

| Col | Meaning |
|---|---|
| Avg, Sta, Dyn | overall + static / dynamic splits |
| VG, DM, SR | 3D Video Grounding · Dimensional Measurement · Spatial Relation |
| R, C, T | Rotational · Counting · Translational |
| FP, SA, DP | False Positives · Speed & Acceleration · Displacement & Path Length |

Reference numbers from this codebase (16 frames):

| Model | # Frames | Avg | Sta | Dyn | VG | DM | SR | R | C | T | FP | SA | DP |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NVILA-Lite-8B (base)  | 16 | 39.0 | 33.0 | 40.8 | 35.9 | 30.1 | 39.0 | 43.4 | 35.7 | 45.2 | 24.4 | 56.8 | 12.5 |
| **4D-RGPT-8B** (ours) | 16 | **46.2** | 41.0 | 47.8 | 44.7 | 37.7 | 46.3 | 50.9 | 46.5 | 46.6 | 49.6 | 59.5 | 45.8 |

> **Note — numbers differ from the paper.** Post-publication, R4D-Bench was
> refreshed (test split re-curated) and the eval pipeline was migrated to
> [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for reproducibility.

---

## Reusing the components in your own model

The three additions drop into any NVILA-shaped MLLM. Each lives in a single file — read the source, the code is short:

### 1. Latent distillation (`L_LD`)

Match a small student projector's per-layer features against a frozen L4P teacher's encoder taps (MSE, per-token).

- Loss + projector + teacher cache: [`perception_distillation()` in llava/model/language_model/llava_llama.py:283-396](llava/model/language_model/llava_llama.py#L283-L396)
- Per-frame hidden-state extraction from the LLM: [`extract_image_region_hidden_states()` at llava/model/language_model/llava_llama.py:227](llava/model/language_model/llava_llama.py#L227)
- Forward-call site (where `l_ld` is added to the SFT loss): [llava/model/language_model/llava_llama.py:204-205](llava/model/language_model/llava_llama.py#L204-L205)

Tuning note: `ld_weight ~ 0.01` in our runs. `hooks_idx` (which teacher layers to distill from) must be the same list in the teacher and the projector — keep it in one config, not two.

### 2. Explicit distillation (`L_ED`)

Decode the student's projected features through the teacher's task heads and match per-task predictions (independently weighted).

- Task loop, depth-log handling, weight parsing: [llava/model/language_model/llava_llama.py:354-390](llava/model/language_model/llava_llama.py#L354-L390)
- CLI arg (`--ed_weights "depth=0.1,flow_2d_backward=0.001,dyn_mask=0.01"`): [llava/train/args.py:81](llava/train/args.py#L81)

Tuning note: natural-scale losses diverge ~100× across tasks (flow is unbounded pixel displacement; depth is bounded log-meters), so equal weighting silently lets flow dominate. Our paper run uses `depth=0.1, flow_2d_backward=0.001, dyn_mask=0.01`.

### 3. Timestamp Positional Encoding (TPE)

Sinusoidal encoding of per-frame timestamps, added to vision tokens before they enter the LLM. Independent of the spatial RoPE / 2D-PE already in the vision encoder. No learned params.

- Module: [llava/model/pe/time_pe.py](llava/model/pe/time_pe.py)
- Wired into the architecture: [llava/model/llava_arch.py:107-108](llava/model/llava_arch.py#L107-L108)
- Applied at the vision-token site (broadcasts over spatial tokens): [llava/model/llava_arch.py:470-476](llava/model/llava_arch.py#L470-L476)
- CLI arg: `--use_time_pe True` — [llava/train/args.py:95](llava/train/args.py#L95)

---

## Running on your own cluster

The `scripts/slurm/` launchers and `scripts/setups/` env helpers are written
for our internal NVIDIA cluster (ADLR / NV SLURM). To adapt them:

- [scripts/slurm/submit.sh](scripts/slurm/submit.sh) + [scripts/slurm/interactive.sh](scripts/slurm/interactive.sh) call the **`submit_job`** CLI (NVIDIA-internal wrapper around `sbatch`). Replace the `submit_job ... --autoresume_* --command ...` invocation with a plain `sbatch` heredoc.
- Partition list **`polar4,polar3,grizzly,polar`** in [scripts/slurm/submit.sh:45](scripts/slurm/submit.sh#L45) — swap for your cluster's partitions.
- **`SLURM_ACCOUNT`** + other env: copy [.env.example](.env.example) to `.env` and fill in.
- The pyxis container image is resolved by [scripts/slurm/_startup.sh](scripts/slurm/_startup.sh) (`resolve_image`) — point this at your own `enroot`/`pyxis` image, or skip pyxis entirely and activate the `.venv` directly in your `sbatch` script.
- The **autoresume callback** at [llava/train/callbacks/autoresume_callback.py](llava/train/callbacks/autoresume_callback.py) targets ADLR's AutoResume SDK. If not on that infra, remove the `AutoResumeCallback(...)` line in [llava/train/train.py](llava/train/train.py) and drop the import.
- **Dataset paths** are now in YAML registries, not in code — `data/registry.yaml` (eval) and `llava/data/registry/datasets/<cluster>.yaml` (train), both gitignored. See [data/README.md](data/README.md).

---

## Dataset Preparation

### R4D-Bench
Please follow our HF dataset instructions here: https://huggingface.co/datasets/nvidia/R4D-Bench.

### Standard 4D/Spatial Benchmarks
Please follow the official instructions of the datasets used in the paper: [STI-Bench](https://mint-sjtu.github.io/STI-Bench.io/), [VLM4D](https://vlm4d.github.io/), [OmniSpatial](https://qizekun.github.io/omnispatial/), [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/), [SAT](https://arijitray.com/SAT/), and [VSTI-Bench](https://vlm-3r.github.io/).

Training mixture for 4D-RGPT-8B: VSTI-Bench (training split), Wolf (NuScenes), RoboFAC, SAT. R4D-Bench is curated from STI-Bench and VLM4D via SoM prompting + human verification. See [data/README.md](data/README.md) for the registry-yaml wiring.

---

## Citation
If you find our work useful, please consider giving a star and citation:
```bibtex
@inproceedings{yang20264d,
  title={4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation},
  author={Yang, Chiao-An and Hachiuma, Ryo and Liu, Sifei and Radhakrishnan, Subhashree and Yeh, Raymond A and Wang, Yu-Chiang Frank and Chen, Min-Hung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={31042--31053},
  year={2026}
}
```

## Licenses

Copyright © 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.
