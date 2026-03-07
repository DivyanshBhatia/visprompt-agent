# VisPrompt

**LLM-Enriched Prompt Fusion for Zero-Shot Vision Models**

> A training-free framework that combines LLM-generated visual descriptions with template ensembles via group-normalized weighted fusion, improving zero-shot classification, retrieval, and action recognition across 10+ datasets and 7 vision backbones.

---

## Key Idea

Foundation vision models like CLIP are powerful but prompt-sensitive: accuracy can swing 10-15% depending on text prompt choice. Prior work uses either hand-crafted template ensembles (Radford et al.) or LLM-generated descriptions (CuPL, DCLIP), but not both optimally.

**VisPrompt** fuses these two signal sources with group-normalized weighting:

```
                    +------------------------+
                    |  80 ImageNet Templates  |---- base_weight / N_templates ----+
                    +------------------------+                                    |
                                                                                  v
    Image --> CLIP --> similarity --> [weighted average] --> prediction
                                                                                  ^
                    +------------------------+                                    |
                    |  LLM Descriptions      |---- desc_weight / N_descriptions --+
                    +------------------------+
```

The key insight: **no fixed weighting works everywhere**. Generic domains (CIFAR-100) need heavy template anchoring (70/30), while fine-grained domains (Flowers, Pets) benefit from description-dominant weights (0/100). The optimal ratio is domain-dependent and dataset-adaptive.

---

## Results Highlights

**Classification** (ViT-L/14, 10 datasets): Wins on 5/10, ties 1, with +3.19% on DTD and +2.21% on Oxford Pets.

**Retrieval** (10 datasets): **10/10 positive improvements**, universal gains from +0.02% to +4.25% mAP.

**Action Recognition** (UCF-101): +3.63% over best baseline, beating all 7 comparison methods.

**Few-shot Comparison**: Our zero-shot method beats 16-shot linear probes on 3/4 datasets (Oxford Pets, CIFAR-100, DTD).

**LLM & Backbone Agnostic**: Consistent gains across 4 LLMs (GPT-4o, GPT-5.2, Claude Sonnet 4, Claude Opus 4.5) and 7 backbones (CLIP, EVA-CLIP, MetaCLIP, SigLIP).

---

## Installation

```bash
git clone https://github.com/your-username/visprompt.git
cd visprompt
pip install -e .
```

### Dependencies

```bash
pip install torch torchvision open-clip-torch
pip install openai        # or: pip install anthropic
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

### Classification (10 datasets)

```bash
# Single dataset with weight ablation
python scripts/run_weight_ablation.py \
    --dataset cifar100 --clip-model ViT-L/14 --llm gpt-4o

# Multi-dataset evaluation with baselines
python scripts/run_multidataset.py \
    --datasets cifar100 flowers102 dtd oxford_pets food101 \
    --clip-model ViT-L/14
```

### Retrieval

```bash
python scripts/run_retrieval_baselines.py \
    --dataset flowers102 --clip-model ViT-L/14 --run-ablation
```

### Action Recognition (UCF-101)

```bash
python scripts/run_action_recognition.py \
    --data-dir /path/to/UCF-101/test --clip-model ViT-L/14
```

### Cross-Product Ablation (LLM x Backbone)

```bash
python scripts/run_cross_ablation.py \
    --datasets cifar100 flowers102 oxford_pets dtd \
    --backbones CLIP-B/32 CLIP-L/14 EVA02-B/16 MetaCLIP-L/14 SigLIP-B/16 \
    --llms GPT-4o GPT-5.2 Claude-Sonnet-4 Claude-Opus-4.5
```

---

## Approach

### 1. LLM Description Generation

A single LLM call generates 10-15 visual descriptions per class:

```
Input:  "Generate visual descriptions for: sunflower, rose, tulip"
Output: {
  "sunflower": [
    "a sunflower, large yellow petals radiating from a dark brown center",
    "a sunflower, tall green stem with a heavy drooping flower head",
    ...
  ]
}
```

### 2. Group-Normalized Weighted Fusion

Templates and descriptions are fused with per-group normalized weights:

```python
# Each template gets:  base_weight / N_templates
# Each description gets: desc_weight / N_descriptions
# This ensures the two groups contribute proportionally regardless of count
```

### 3. Weight Selection

The optimal base/description weight ratio varies by domain:

| Domain | Optimal Weight | Why |
|--------|---------------|-----|
| Generic (CIFAR-100) | 70/30 | Templates provide stable generic recognition |
| Textures (DTD) | 40/60 | Texture descriptions add discriminative cues |
| Fine-grained (Flowers, Pets) | 0/100 | Descriptions dominate -- visual details matter most |
| Actions (UCF-101) | 55/45 | Balanced -- actions need both context and specifics |

---

## Experiments

### Supported Datasets

cifar10, cifar100, flowers102, dtd, food101, oxford_pets, caltech101, eurosat, fgvc_aircraft, country211, ucf101

### Supported Backbones

| Family | Models |
|--------|--------|
| OpenAI CLIP | ViT-B/32, ViT-B/16, ViT-L/14 |
| EVA-CLIP | EVA02-B-16 |
| MetaCLIP | ViT-B-32, ViT-L-14 |
| SigLIP | ViT-B-16 |

### Supported LLMs

GPT-4o, GPT-5.2, Claude Sonnet 4, Claude Opus 4.5

### Run All Experiments

```bash
# Classification + baselines (Table 1)
python scripts/run_baselines.py --dataset cifar100 --clip-model ViT-L/14

# Weight ablation (Figure 2)
python scripts/run_weight_ablation.py --dataset flowers102 --clip-model ViT-L/14

# Cross-product ablation (Table 2)
python scripts/run_cross_ablation.py --datasets cifar100 flowers102 oxford_pets dtd

# Few-shot comparison (Figure 3)
python scripts/run_fewshot_comparison.py --datasets flowers102 dtd oxford_pets cifar100

# Variance analysis (Table 3)
python scripts/run_variance.py --dataset cifar100 --n-trials 5

# Description scaling (Figure 4)
python scripts/run_desc_scaling.py --dataset flowers102 --clip-model ViT-L/14

# Retrieval (Table 4)
python scripts/run_retrieval_baselines.py --dataset flowers102 --run-ablation

# Segmentation (Table 5)
python scripts/run_seg_baselines.py

# Action recognition (Table 6)
python scripts/run_action_recognition.py --data-dir /path/to/UCF-101/test
```

---

## Project Structure

```
visprompt/
├── visprompt/
│   ├── __init__.py
│   ├── baselines/
│   │   └── __init__.py          # Baseline methods (CuPL, WaffleCLIP, DCLIP, etc.)
│   ├── tasks/
│   │   ├── base.py              # BaseTaskRunner interface
│   │   ├── classification.py    # CLIP zero-shot classification
│   │   └── segmentation_clipseg.py  # CLIPSeg segmentation
│   └── utils/
│       ├── llm.py               # LLM client (OpenAI, Anthropic) + cost tracking
│       └── metrics.py           # Metrics (accuracy, mAP, IoU, per-class)
├── scripts/
│   ├── run_weight_ablation.py   # Weight ratio sweep (core method)
│   ├── run_baselines.py         # Baseline comparison (9 methods)
│   ├── run_cross_ablation.py    # LLM x backbone cross-product
│   ├── run_retrieval_baselines.py   # Zero-shot retrieval
│   ├── run_fewshot_comparison.py    # Few-shot linear probe comparison
│   ├── run_variance.py          # Multi-trial variance analysis
│   ├── run_desc_scaling.py      # Description count scaling
│   ├── run_action_recognition.py    # UCF-101 action recognition
│   ├── run_seg_baselines.py     # Segmentation baselines
│   └── run_multidataset.py      # Multi-dataset evaluation
├── experiments/                 # Output directory for results
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Cost

LLM description generation is inexpensive:

| Dataset | Classes | LLM | Cost |
|---------|---------|-----|------|
| CIFAR-100 | 100 | GPT-4o | ~$0.20 |
| Flowers102 | 102 | GPT-4o | ~$0.23 |
| DTD | 47 | GPT-4o | ~$0.10 |
| Oxford Pets | 37 | GPT-4o | ~$0.09 |

Total cost for all 10 datasets: **< $2.00** with GPT-4o.

---

## Citation

```bibtex
@inproceedings{visprompt2026,
  title={LLM-Enriched Prompt Fusion for Zero-Shot Vision Models},
  author={},
  booktitle={},
  year={2026}
}
```

---

## License

MIT
