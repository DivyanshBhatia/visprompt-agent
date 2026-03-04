# VisPromptAgent — Experiment Plan

> Run these experiments in order. Each step builds on the previous one.
> After each step, share the output JSON / numbers with me and I'll help analyze results and draft the corresponding paper section.

---

## Environment Setup (Do This First)

```bash
# Clone and install
cd visprompt-agent
pip install -e ".[clip]"
pip install openai torchvision

# Verify
python -c "import visprompt; print(visprompt.__version__)"
python -c "import open_clip; print('open_clip OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Set API key
export OPENAI_API_KEY="sk-..."
```

**Hardware needed:** 1× GPU with ≥8GB VRAM (for CLIP). LLM calls go to API.

---

## Phase 1 — Classification (CIFAR-100)

This produces Tables 2, 3, and 5. Classification is the fastest task to iterate on.

---

### Experiment 1.1 — Baselines (Table 2, rows 1–4)

**Purpose:** Establish the comparison floor. No LLM calls needed for most of these.

```bash
python scripts/run_baselines.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --device cuda \
    --val-size 10000 \
    --output-dir experiments/baselines
```

**What this runs:**
- Single template: `"a photo of a {class}"` (CLIP baseline)
- 80-template ensemble (Radford et al. 2021)
- WaffleCLIP (Roth et al. 2023) — random descriptor ensemble
- CuPL (Pratt et al. 2023) — needs OpenAI API

**Share with me:** `experiments/baselines/cifar100_baselines.json`

**Expected runtime:** ~10 min (CLIP inference) + ~2 min (CuPL API calls)

---

### Experiment 1.2 — VisPromptAgent Full System (Table 2, last rows)

**Purpose:** Our main result on CIFAR-100 with ViT-B/32.

```bash
python scripts/run.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm gpt-4o \
    --llm-provider openai \
    --max-iter 3 \
    --val-size 10000 \
    --device cuda \
    --output-dir experiments/main \
    --verbose
```

**Share with me:**
1. `experiments/main/cifar100_classification_*/summary.json`
2. `experiments/main/cifar100_classification_*/reasoning_trace.json` (the full agent reasoning — I need this to write Section 4.4 qualitative analysis)
3. `experiments/main/cifar100_classification_*/cost.json`

**Expected runtime:** ~20 min total (~7 min CLIP per iteration + ~2 min LLM per iteration × 3)

---

### Experiment 1.3 — VisPromptAgent with ViT-L/14 (Table 2, last row)

**Purpose:** Show scaling to a stronger backbone.

```bash
python scripts/run.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-L/14 \
    --llm gpt-4o \
    --max-iter 3 \
    --val-size 10000 \
    --device cuda \
    --output-dir experiments/main_vitl
```

**Share with me:** `experiments/main_vitl/cifar100_classification_*/summary.json`

---

### Experiment 1.4 — Agent Ablation (Table 3, top section)

**Purpose:** Show every agent contributes. This is the most important table for convincing reviewers.

```bash
python scripts/run_ablation.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm gpt-4o \
    --max-iter 3 \
    --val-size 10000 \
    --ablations agent \
    --output-dir experiments/ablation \
    --verbose
```

**What this runs (5 configurations):**

| Config | Description |
|--------|-------------|
| `full` | All 5 agents, 3 iterations |
| `no_analyst` | Planner gets no dataset intelligence |
| `no_critic` | No unit tests, no quality feedback |
| `no_strategist` | Single iteration only |
| `single_agent` | One LLM call, no role decomposition |

**Share with me:** `experiments/ablation/all_results.json`

**Expected runtime:** ~1.5 hours (5 × full pipeline runs)

---

### Experiment 1.5 — Iteration Ablation (Table 3, footnote / Figure)

**Purpose:** Show diminishing returns curve (justifies stopping at 3).

```bash
python scripts/run_ablation.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm gpt-4o \
    --ablations iteration \
    --val-size 10000 \
    --output-dir experiments/ablation
```

**What this runs:** 1, 2, 3, and 5 iterations.

**Share with me:** the iteration section from `experiments/ablation/all_results.json`

---

### Experiment 1.6 — VLM Backbone Ablation (Table 3, bottom section)

**Purpose:** Show it works with different LLMs, including cheaper/open ones.

```bash
# GPT-4o (already done in 1.2)

# GPT-4o-mini (cheap alternative)
python scripts/run.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm gpt-4o-mini \
    --max-iter 3 \
    --val-size 10000 \
    --output-dir experiments/backbone_4omini

# Claude Sonnet (if you have Anthropic key)
python scripts/run.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm claude-sonnet-4-20250514 \
    --llm-provider anthropic \
    --max-iter 3 \
    --val-size 10000 \
    --output-dir experiments/backbone_claude
```

**Share with me:** `summary.json` from each run

---

## Phase 2 — Segmentation (DAVIS 2017)

This is for Table 4 (cross-task evaluation). Same 5-agent architecture, different task.

---

### Experiment 2.1 — SAM 2 Segmentation

**Pre-requisites:**
1. Download DAVIS 2017 val set: https://davischallenge.org/davis2017/code.html
2. Download SAM 2 checkpoint: `sam2_hiera_base_plus.pt`

```bash
# Install SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

python scripts/run.py \
    --task segmentation \
    --dataset davis2017 \
    --sam-checkpoint /path/to/sam2_hiera_base_plus.pt \
    --sam-model-type hiera_b+ \
    --data-dir /path/to/DAVIS/JPEGImages/480p \
    --annotation-dir /path/to/DAVIS/Annotations/480p \
    --llm gpt-4o \
    --max-iter 3 \
    --device cuda \
    --output-dir experiments/segmentation
```

**Share with me:** `summary.json` + `reasoning_trace.json`

**Key thing to report:** The Critic's auto-generated tests should look DIFFERENT from classification — IoU thresholds, boundary quality, multi-object consistency instead of accuracy/confusion pairs. Screenshot the critic's output from the reasoning trace.

---

### Experiment 2.2 — SAM 2 Baselines

We need baseline SAM performance to compare against:

```python
# Quick script — run this manually:
# experiments/sam_baselines.py

import numpy as np
from visprompt.tasks.segmentation import SAMSegmentationRunner
from visprompt.agents.base import TaskSpec

task_spec = TaskSpec(
    task_type="segmentation",
    dataset_name="davis2017",
    foundation_model="sam2",
    prompt_modality="point",
    metric_name="J&F",
)

runner = SAMSegmentationRunner(
    sam_checkpoint="/path/to/sam2_hiera_base_plus.pt",
    model_type="hiera_b+",
    device="cuda",
    image_dir="/path/to/DAVIS/JPEGImages/480p",
    annotation_dir="/path/to/DAVIS/Annotations/480p",
)
runner.load_data()

# Baseline 1: Center point
prompts_center = {
    "type": "segmentation",
    "point_strategy": {"initial_points": 1, "placement": "center", "negative_points": False},
    "box_strategy": {},
    "text_prompts": {},
    "multi_object_handling": "per_instance",
}
result = runner.evaluate(prompts_center, task_spec)
print(f"Center point: {result.primary_metric:.4f}")

# Baseline 2: 5-point grid
prompts_grid = {
    "type": "segmentation",
    "point_strategy": {"initial_points": 5, "placement": "grid", "negative_points": False},
    "box_strategy": {},
    "text_prompts": {},
    "multi_object_handling": "per_instance",
}
result = runner.evaluate(prompts_grid, task_spec)
print(f"5-point grid: {result.primary_metric:.4f}")

# Baseline 3: GT box (oracle)
prompts_box = {
    "type": "segmentation",
    "point_strategy": {},
    "box_strategy": {"source": "ground_truth", "padding_ratio": 0.1},
    "text_prompts": {},
    "multi_object_handling": "per_instance",
}
result = runner.evaluate(prompts_box, task_spec)
print(f"GT box: {result.primary_metric:.4f}")
```

**Share with me:** The three numbers.

---

## Phase 3 — Detection (LVIS rare)

This is the third task for Table 4.

---

### Experiment 3.1 — GroundingDINO Detection

**Pre-requisites:**
1. Download LVIS val annotations: https://www.lvisdataset.org/dataset
2. Download COCO val2017 images (LVIS uses same images)
3. Download GroundingDINO checkpoint

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

python scripts/run.py \
    --task detection \
    --dataset lvis \
    --gdino-config /path/to/GroundingDINO_SwinB_cfg.py \
    --gdino-checkpoint /path/to/groundingdino_swinb_cogcoor.pth \
    --data-dir /path/to/coco/val2017 \
    --annotation-file /path/to/lvis_v1_val.json \
    --llm gpt-4o \
    --max-iter 3 \
    --device cuda \
    --output-dir experiments/detection
```

**Share with me:** `summary.json` + `reasoning_trace.json`

---

## Phase 4 — Additional Datasets (Strengthens Paper)

If time permits, these add extra rows to Table 4 and make the generality argument airtight.

### 4.1 — ImageNet (stronger classification benchmark)

```bash
python scripts/run.py \
    --task classification \
    --config configs/tasks/imagenet.yaml \
    --data-dir /path/to/imagenet/val \
    --clip-model ViT-B/32 \
    --llm gpt-4o \
    --max-iter 3 \
    --output-dir experiments/imagenet
```

You'll need to create `configs/tasks/imagenet.yaml` — I can help with that once you have the data.

### 4.2 — EuroSAT or DTD (domain shift evaluation)

Shows VisPromptAgent adapts to non-natural domains.

---

## What to Share After Each Experiment

For every experiment, I need:

| File | Why I Need It |
|------|--------------|
| `summary.json` | Numbers for tables (metric, iterations, cost) |
| `reasoning_trace.json` | Qualitative analysis — agent reasoning examples for Section 4.4 |
| `cost.json` | Per-agent cost breakdown for Table 5 |
| Terminal output | Any errors or warnings to debug |

For the **first run** (Experiment 1.2), also share:
- The **analyst report** (`analyst_report.json`) — I'll use it to write the walkthrough example in the paper
- The **Critic's test output** from each iteration — I'll showcase how tests evolve

---

## Priority Order

If you're short on time, here's what matters most:

| Priority | Experiment | Paper Contribution |
|----------|-----------|-------------------|
| **P0** | 1.1 + 1.2 | Table 2 (main result + baselines) |
| **P0** | 1.4 | Table 3 (ablation — reviewers will reject without this) |
| **P1** | 1.5 | Iteration curve (supports Table 3) |
| **P1** | 2.1 + 2.2 | Table 4 row 2 (cross-task) |
| **P1** | 3.1 | Table 4 row 3 (cross-task) |
| **P2** | 1.3 | Table 2 last row (backbone scaling) |
| **P2** | 1.6 | Table 3 bottom (VLM backbone) |
| **P3** | 4.x | Extra datasets (bonus rows) |

Start with P0 experiments — those alone give us a submittable paper.

---

## Expected Cost Budget

| Experiment | Est. API Cost | Est. GPU Time |
|-----------|--------------|--------------|
| 1.1 Baselines | ~$0.50 (CuPL only) | ~10 min |
| 1.2 Full system | ~$3–5 | ~20 min |
| 1.3 ViT-L/14 | ~$3–5 | ~30 min |
| 1.4 Agent ablation | ~$15–20 | ~1.5 hr |
| 1.5 Iteration ablation | ~$8–10 | ~1 hr |
| 1.6 Backbone ablation | ~$5–8 per model | ~20 min each |
| 2.x Segmentation | ~$5–8 | ~30 min |
| 3.x Detection | ~$5–8 | ~40 min |
| **Total (P0+P1)** | **~$40–60** | **~4 hours** |
