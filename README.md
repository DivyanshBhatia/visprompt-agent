# NETRA

**Normalized Ensembling of Textual Representations for Zero-Shot Visual Adaptation**

> A training-free framework that fuses LLM-generated visual descriptions with hand-crafted template ensembles via group-normalized weighted fusion. No training, no target images, no per-image LLM calls — just better text prompts for CLIP.

📄 **Paper**: BMVC 2026 submission  
💻 **Code**: This repository

---

## Key Idea

CLIP's zero-shot accuracy depends heavily on text prompt design. Templates like "a photo of a {class}" provide stability but lack discriminative power. LLM-generated descriptions provide fine-grained detail but can be noisy. **NETRA fuses both via group-normalized weighting:**

```
t_c = (α/M) · Σ f̂(templates) + (β/N) · Σ f̂(descriptions)
```

Group normalization decouples influence from group size: 10 descriptions can contribute as much as 80 templates when α/β is set appropriately. A single fixed default of **55/45** improves over templates on **8/10 datasets** without any tuning.

---

## Results

### Classification (ViT-L/14, 10 datasets, 9 baselines)

| Dataset | Templates | Best Baseline | **NETRA** | Δ |
|---------|-----------|--------------|-----------|---|
| DTD | 52.82% | 54.47% (CuPL+e) | **57.66%** | **+3.19%** |
| Oxford Pets | 90.16% | 90.27% (CuPL+e) | **92.48%** | **+2.21%** |
| Flowers102 | 68.09% | 74.21% (DCLIP) | **75.54%** | **+1.33%** |
| CIFAR-100 | 74.70% | 75.71% (CLIP-Enh.) | **75.81%** | **+0.10%** |
| Country211 | 22.13% | 23.84% (WaffleCLIP) | **24.20%** | **+0.36%** |

Wins on 5/10 datasets, ties on 1, competitive on remaining 4.

### Retrieval (10 datasets)

**10/10 positive improvements** over template ensembles, from +0.02% to +4.25% mAP. EuroSAT R@1 improves by +10%.

### Action Recognition (UCF-101)

**+3.63% over all 8 baselines** (75.88% vs 72.48% best baseline), using middle-frame extraction.

### Zero-Shot vs Few-Shot

Our zero-shot method **beats 16-shot linear probes on 3/4 datasets** (Oxford Pets, CIFAR-100, DTD) without any labeled data.

### vs TLAC (Per-Image LMM, CVPR 2025W)

| | NETRA | TLAC (GPT-4o vision) |
|---|---|---|
| Total cost (11 datasets) | **$1.95** | $77.75 |
| Cost ratio | — | **40× more** |
| NETRA wins | 2 (CIFAR-10, CIFAR-100) | — |
| Within 1% | 3 (Food, Pets, Caltech) | — |
| TLAC wins (>1%) | — | 6 |
| Works on low-res (32×32) | ✅ | ❌ (−15.7% on CIFAR-100) |
| Works for retrieval | ✅ (10/10) | ❌ |
| Per-image inference cost | **$0** | $0.0004–0.002 |

### Cross-Backbone & Cross-LLM

Tested across **7 backbones** (OpenAI CLIP, EVA-CLIP, MetaCLIP, SigLIP) × **4 LLMs** (GPT-4o, GPT-5.2, Claude Sonnet 4, Claude Opus 4.5) = **112 positive results** on 4 datasets. LLM ranking is domain-dependent: Opus 4.5 dominates fine-grained; GPT-5.2 dominates textures.

---

## Installation

```bash
git clone https://github.com/DivyanshBhatia/visprompt-agent.git
cd visprompt-agent
pip install -e .
```

### Dependencies

```bash
pip install torch torchvision open-clip-torch
pip install openai anthropic  # for LLM description generation
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
# and/or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

### Classification with baselines

```bash
python scripts/run_baselines.py --dataset flowers102 --clip-model ViT-L/14
```

### Weight ablation (core method)

```bash
python scripts/run_weight_ablation.py --dataset dtd --clip-model ViT-L/14 --llm gpt-4o
```

### Multi-dataset evaluation

```bash
python scripts/run_multidataset.py \
    --datasets cifar100 flowers102 dtd oxford_pets food101 \
    --clip-model ViT-L/14
```

---

## All Experiments

### Paper Tables & Figures

| Paper Reference | Script | Command |
|----------------|--------|---------|
| Table 2: Classification | `run_baselines.py` | `python scripts/run_baselines.py --dataset cifar100` |
| Table 3: Retrieval | `run_retrieval_baselines.py` | `python scripts/run_retrieval_baselines.py --dataset flowers102 --run-ablation` |
| Table 4: Action Recognition | `run_action_recognition.py` | `python scripts/run_action_recognition.py --data-dir /path/to/UCF-101/test` |
| Table 5: Cross-LLM×Backbone | `run_cross_ablation.py` | `python scripts/run_cross_ablation.py --datasets cifar100 flowers102` |
| Table 6: Few-shot comparison | `run_fewshot_comparison.py` | `python scripts/run_fewshot_comparison.py --datasets flowers102 dtd` |
| Table 7: Variance | `run_variance.py` | `python scripts/run_variance.py --dataset cifar100 --n-trials 5` |
| Table 8: Fixed default | `run_weight_ablation.py` | Use `--base-weight 0.55` across all datasets |
| Table 9: TLAC comparison | `run_tlac_baseline.py` | `python scripts/run_tlac_baseline.py --dataset dtd --mode tlac` |
| Fig 2: Weight ablation | `run_weight_ablation.py` | Run per dataset, plot results |
| Fig 4: Description scaling | `run_desc_scaling.py` | `python scripts/run_desc_scaling.py --dataset flowers102` |
| Segmentation | `run_seg_baselines.py` | `python scripts/run_seg_baselines.py` |

### TLAC Baseline (Per-Image LMM)

```bash
# Requires OPENAI_API_KEY with GPT-4o vision access
# Cost: ~$0.001/image with GPT-4o

python scripts/run_tlac_baseline.py --dataset dtd --clip-model ViT-L/14 --mode tlac
python scripts/run_tlac_baseline.py --dataset oxford_pets --clip-model ViT-L/14 --mode tlac
python scripts/run_tlac_baseline.py --dataset flowers102 --clip-model ViT-L/14 --mode tlac
python scripts/run_tlac_baseline.py --dataset cifar100 --clip-model ViT-L/14 --mode tlac
python scripts/run_tlac_baseline.py --dataset ucf101 --clip-model ViT-L/14 --mode tlac \
    --data-dir /path/to/ucf101/test
```

### Action Recognition (UCF-101)

```bash
# Full run with all baselines
python scripts/run_action_recognition.py \
    --data-dir /path/to/UCF-101/test --clip-model ViT-L/14

# Ablation only (skip baselines)
python scripts/run_action_ablation.py \
    --data-dir /path/to/UCF-101/test --clip-model ViT-L/14
```

---

## Method

### Algorithm

```
OFFLINE (once per dataset, ~$0.10-0.25):
  For each class c:
    T_c ← 80 hand-crafted ImageNet templates
    D_c ← LLM generates 10-15 visual descriptions
    t_c ← (α/M) · Σ f̂_T(templates) + (β/N) · Σ f̂_T(descriptions)
    t̄_c ← normalize(t_c)

ONLINE (per image, zero additional cost):
    ŷ ← argmax_c cos(f_I(x), t̄_c)    # standard CLIP inference
```

### Weight Selection Guide

| Domain Type | Recommended α/β | Examples |
|-------------|-----------------|----------|
| Generic objects | 55/45 – 70/30 | CIFAR, Caltech, ImageNet |
| Fine-grained | 0/100 – 20/80 | Flowers, Pets, Birds |
| Textures | 40/60 | DTD, materials |
| Food / scenes | 55/45 | Food101, SUN397 |
| Unknown | **55/45** | Safe default, works on 8/10 datasets |

### Description Scaling

- **With template anchoring** (α > 0): more descriptions monotonically help; plateau at 6-8
- **Without anchoring** (α = 0): peak at 2-3 descriptions, then declines
- **Practical recommendation**: 5-8 descriptions per class with α ≥ 0.4

---

## Supported Configurations

### Datasets (11)

`cifar10`, `cifar100`, `flowers102`, `dtd`, `food101`, `oxford_pets`, `caltech101`, `eurosat`, `fgvc_aircraft`, `country211`, `ucf101`

### Backbones (7 from 4 families)

| Family | Models | Pretrained |
|--------|--------|------------|
| OpenAI CLIP | ViT-B/32, ViT-B/16, ViT-L/14 | `openai` |
| EVA-CLIP | EVA02-B-16 | `merged2b_s8b_b131k` |
| MetaCLIP | ViT-B-32-quickgelu, ViT-L-14-quickgelu | `metaclip_400m` |
| SigLIP | ViT-B-16-SigLIP | `webli` |

### LLMs (4)

| LLM | Provider | Best for |
|-----|----------|----------|
| GPT-4o | OpenAI | Cost-efficient, generic domains |
| GPT-5.2 | OpenAI | Textures (DTD) |
| Claude Sonnet 4 | Anthropic | Balanced |
| Claude Opus 4.5 | Anthropic | Fine-grained (Flowers, Pets) |

---

## Project Structure

```
visprompt-agent/
├── visprompt/
│   ├── __init__.py
│   ├── task_spec.py              # TaskSpec dataclass
│   ├── baselines/
│   │   └── __init__.py           # CuPL, WaffleCLIP, DCLIP, CLIP-Enhance, Frolic
│   ├── tasks/
│   │   ├── base.py               # BaseTaskRunner interface
│   │   ├── classification.py     # CLIP classification (TTA, logit ensemble)
│   │   ├── segmentation_clipseg.py
│   │   └── detection.py
│   └── utils/
│       ├── llm.py                # Multi-provider LLM client + cost tracking
│       └── metrics.py            # Accuracy, mAP, IoU, per-class metrics
├── scripts/
│   ├── run.py                    # Core: build_task_spec, build_task_runner
│   ├── run_baselines.py          # 9 classification baselines
│   ├── run_weight_ablation.py    # α/β weight sweep
│   ├── run_cross_ablation.py     # LLM × backbone cross-product
│   ├── run_retrieval_baselines.py
│   ├── run_fewshot_comparison.py
│   ├── run_variance.py           # Multi-trial robustness
│   ├── run_desc_scaling.py       # Description count scaling
│   ├── run_action_recognition.py # UCF-101 (full with baselines)
│   ├── run_action_ablation.py    # UCF-101 (ablation only)
│   ├── run_seg_baselines.py      # Pascal VOC segmentation
│   ├── run_multidataset.py       # Multi-dataset batch evaluation
│   ├── run_tlac_baseline.py      # TLAC per-image LMM baseline
│   ├── run_protext_baseline.py   # ProText published numbers
│   └── compare_2025_methods.py   # 2025 method comparison table
├── experiments/                   # Output JSON results
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Cost

### NETRA (per-class, one-time)

| Dataset | Classes | GPT-4o Cost | Reusable? |
|---------|---------|-------------|-----------|
| CIFAR-100 | 100 | ~$0.20 | ✅ Forever |
| Flowers102 | 102 | ~$0.25 | ✅ Forever |
| DTD | 47 | ~$0.09 | ✅ Forever |
| All 11 datasets | — | **~$1.95** | ✅ Forever |

### TLAC comparison (per-image, every run)

| | NETRA | TLAC |
|---|---|---|
| 11 datasets | **$1.95** | **$77.75** |
| 1M images | **$1.95** | **~$1,600** |
| Cost scaling | O(classes) | O(images) |

---

## Citation

```bibtex
@inproceedings{netra2026,
  title={NETRA: Normalized Ensembling of Textual Representations for Zero-Shot Visual Adaptation},
  author={Anonymous},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2026}
}
```

---

## License

MIT
