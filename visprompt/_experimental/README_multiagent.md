# VisPromptAgent

**Multi-Agent Visual Prompt Engineering for Foundation Vision Models**

> A training-free, dataset-generic framework where 5 specialized LLM agents collaborate to iteratively optimize visual prompts for CLIP, SAM 2, GroundingDINO, and other foundation models — across classification, segmentation, and detection tasks.

---

## Key Idea

Foundation vision models are powerful but prompt-brittle: CLIP accuracy swings 10–15% depending on text prompt choice, SAM segmentation quality varies drastically with point placement, and GroundingDINO detection depends heavily on text descriptions. VisPromptAgent closes this gap automatically.

Instead of a single LLM call or manual engineering, VisPromptAgent decomposes prompt optimization into 5 specialized agents that collaborate through an iterative refinement loop:

```
Dataset Analyst → Prompt Planner → Prompt Executor → Quality Critic → Refinement Strategist
       ↑                                                                        │
       └────────────────────────── iterate ──────────────────────────────────────┘
```

The same 5-agent architecture works across tasks — only the Executor (which foundation model to call) and the Critic's auto-generated tests change.

---

## Architecture

| Agent | Role | Why It Matters |
|-------|------|----------------|
| **Dataset Analyst** | Examines dataset structure, predicts confusion pairs via text-embedding similarity, provides domain insights | Without it, the Planner generates generic prompts. Removing it costs ~3.3% accuracy. |
| **Prompt Planner** | Designs hierarchical prompt strategies (base templates → superclass context → confusion-pair discriminators) | Allocates "prompt budget" to the hardest cases. Plan/execute separation prevents satisficing. |
| **Prompt Executor** | Runs the foundation model, returns rich diagnostics (per-class metrics, confusion matrix, confidence distributions) | Intentionally simple — translates strategy to prompts and executes. |
| **Quality Critic** | **Auto-generates** task-appropriate unit tests at runtime using LLM reasoning, then runs them | **Key differentiator.** No hand-written tests. Different tasks get different tests automatically. |
| **Refinement Strategist** | Diagnoses root causes (prompt issue vs. resolution limit vs. model gap), proposes targeted fixes | Single-agent systems randomly retry. The Strategist diagnoses and prescribes specifically. |

### Auto-Generated Visual Unit Tests

The Critic generates tests appropriate to each task:

- **Classification**: accuracy thresholds, per-class minimums, confusion symmetry, confidence calibration, resolution impact
- **Segmentation**: IoU thresholds, boundary quality, multi-object consistency, prompt efficiency
- **Detection**: AP by object size, false positive rates, rare class recall, confidence distributions

Tests evolve across iterations — new tests emerge as patterns become visible (e.g., a "diminishing returns" test appears after iteration 2).

---

## Installation

```bash
git clone https://github.com/your-username/visprompt-agent.git
cd visprompt-agent
pip install -e .
```

### Task-specific dependencies

```bash
# Classification (CLIP)
pip install torch torchvision open-clip-torch

# Segmentation (SAM 2)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Detection (GroundingDINO)
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### LLM providers

```bash
# At least one required:
pip install openai        # GPT-4o (default)
pip install anthropic     # Claude
pip install google-generativeai  # Gemini
```

Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

### CIFAR-100 Classification

```bash
python scripts/run.py \
    --task classification \
    --dataset cifar100 \
    --clip-model ViT-B/32 \
    --llm gpt-4o \
    --max-iter 3
```

### Python API

```python
from visprompt import VisPromptPipeline
from visprompt.agents import TaskSpec
from visprompt.tasks import CLIPClassificationRunner

# 1. Define your task
task_spec = TaskSpec(
    task_type="classification",
    dataset_name="cifar100",
    class_names=["apple", "aquarium_fish", ...],  # your classes
    num_classes=100,
    image_resolution=(32, 32),
    foundation_model="clip",
    prompt_modality="text",
    metric_name="top1_accuracy",
)

# 2. Set up the foundation model runner
runner = CLIPClassificationRunner(
    clip_model_name="ViT-B/32",
    device="cuda",
    images=your_images,      # (N, H, W, C) numpy array
    labels=your_labels,      # (N,) numpy array
)

# 3. Run the pipeline
pipeline = VisPromptPipeline(
    task_spec=task_spec,
    task_runner=runner,
    llm_model="gpt-4o",
)
result = pipeline.run(max_iterations=3)

# 4. Inspect results
print(result.summary())
print(f"Cost: ${result.cost_summary['total_cost_usd']:.2f}")
print(f"Per-agent costs: {result.cost_summary['per_agent']}")
```

### Custom Dataset

```python
task_spec = TaskSpec(
    task_type="classification",
    dataset_name="my_dataset",
    class_names=["cat", "dog", "bird"],
    num_classes=3,
    image_resolution=(224, 224),
    domain="natural",           # or "medical", "remote_sensing", etc.
    foundation_model="clip",
    prompt_modality="text",
    metric_name="top1_accuracy",
)
```

Or use a YAML config:

```yaml
# configs/tasks/my_dataset.yaml
task_type: classification
dataset_name: my_dataset
class_names: [cat, dog, bird]
num_classes: 3
image_resolution: [224, 224]
domain: natural
foundation_model: clip
prompt_modality: text
metric_name: top1_accuracy
```

```bash
python scripts/run.py --task classification --config configs/tasks/my_dataset.yaml \
    --data-dir /path/to/images
```

---

## Experiments

### Run Baselines (Table 2)

```bash
python scripts/run_baselines.py --task classification --dataset cifar100
```

Runs all baseline methods:
- Single template (`"a photo of a {class}"`)
- 80-template ensemble (Radford et al. 2021)
- CuPL (Pratt et al. 2023)
- WaffleCLIP (Roth et al. 2023)

### Run Ablation Study (Table 3)

```bash
python scripts/run_ablation.py --task classification --dataset cifar100 \
    --ablations agent iteration backbone
```

Ablation dimensions:
- **Agent ablation**: Full system, w/o Analyst, w/o Critic, w/o Strategist, single agent
- **Iteration ablation**: 1, 2, 3, 5 iterations
- **Backbone ablation**: GPT-4o, GPT-4o-mini, Claude Sonnet, Qwen2-VL

### Cross-Task Evaluation (Table 4)

```bash
# Classification
python scripts/run.py --task classification --dataset cifar100

# Segmentation
python scripts/run.py --task segmentation --dataset davis2017 \
    --sam-checkpoint sam2_vit_b.pth --data-dir /data/davis

# Detection
python scripts/run.py --task detection --dataset lvis \
    --gdino-config GroundingDINO_SwinB.py --gdino-checkpoint gdino.pth \
    --data-dir /data/lvis/val --annotation-file /data/lvis/lvis_val.json
```

---

## Project Structure

```
visprompt-agent/
├── visprompt/
│   ├── __init__.py              # Package entry point
│   ├── pipeline.py              # Main orchestrator + AblationPipeline
│   ├── agents/
│   │   ├── base.py              # BaseAgent, TaskSpec, AgentMessage
│   │   ├── analyst.py           # Dataset Analyst
│   │   ├── planner.py           # Prompt Planner (task-specific strategies)
│   │   ├── executor.py          # Prompt Executor (model inference)
│   │   ├── critic.py            # Quality Critic (auto-generated tests)
│   │   └── strategist.py        # Refinement Strategist (root-cause diagnosis)
│   ├── tasks/
│   │   ├── base.py              # BaseTaskRunner interface
│   │   ├── classification.py    # CLIP zero-shot classification
│   │   ├── segmentation.py      # SAM/SAM2 interactive segmentation
│   │   └── detection.py         # GroundingDINO open-vocab detection
│   ├── models/
│   │   └── __init__.py          # Model loading utilities
│   ├── tests/
│   │   └── __init__.py          # Deterministic test framework (for ablation)
│   ├── baselines/
│   │   └── __init__.py          # Baseline methods (Table 2)
│   └── utils/
│       ├── llm.py               # LLM client + cost tracking
│       └── metrics.py           # Task-agnostic metrics (accuracy, IoU, AP)
├── scripts/
│   ├── run.py                   # Main entry point
│   ├── run_baselines.py         # Baseline comparison
│   └── run_ablation.py          # Ablation experiments
├── configs/
│   ├── default.yaml
│   └── tasks/
│       ├── classification.yaml
│       ├── segmentation.yaml
│       └── detection.yaml
├── experiments/                 # Output directory for results
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Extending VisPromptAgent

### Adding a New Task

1. Create a new `TaskRunner` in `visprompt/tasks/`:

```python
from visprompt.tasks.base import BaseTaskRunner
from visprompt.utils.metrics import EvalResult

class MyTaskRunner(BaseTaskRunner):
    def load_data(self, split="val"):
        # Load your dataset
        ...

    def evaluate(self, prompts, task_spec):
        # Run your foundation model with the given prompts
        # Return an EvalResult
        return EvalResult(
            primary_metric=0.75,
            primary_metric_name="my_metric",
            per_class_metrics={"class_a": 0.8, "class_b": 0.7},
        )
```

2. Define a `TaskSpec`:

```python
task_spec = TaskSpec(
    task_type="my_task",
    dataset_name="my_dataset",
    class_names=["class_a", "class_b"],
    num_classes=2,
    foundation_model="my_model",
    prompt_modality="text",
    metric_name="my_metric",
)
```

3. Add task-specific prompts in the Planner by adding a new system prompt in `planner.py`:

```python
PLANNER_SYSTEMS["my_task"] = """Your system prompt for planning..."""
```

The Critic and Strategist will automatically adapt — the Critic generates tests based on whatever metrics the Executor returns.

### Adding a New LLM Provider

Extend `LLMClient._init_client()` and add a `_call_<provider>()` method in `visprompt/utils/llm.py`.

### Adding a New Baseline

Add a method to `BaselineRunner` in `visprompt/baselines/__init__.py`.

---

## Cost Tracking

Every LLM call is tracked with per-agent attribution:

```python
result = pipeline.run(max_iterations=3)
print(result.cost_summary)
# {
#   "total_cost_usd": 3.36,
#   "total_input_tokens": 336000,
#   "total_output_tokens": 84000,
#   "total_calls": 84,
#   "per_agent": {
#     "dataset_analyst": 0.72,
#     "prompt_planner": 0.48,
#     "quality_critic": 0.96,
#     "refinement_strategist": 1.20,
#   }
# }
```

---

## Design Decisions

These decisions are informed by the analysis of AutoKaggle's ICLR 2025 rejection:

| Decision | Rationale |
|----------|-----------|
| Auto-generated tests (not hand-written) | Eliminates fairness concerns — the Critic reasons about what to test, not the authors |
| 5 specialized agents (not 1 general agent) | Ablation shows single→multi gains 4.7%; each agent contributes 2.5–3.5% |
| 8+ baselines from day one | AutoKaggle launched with 0 baselines; all 3 reviewers flagged this |
| Cross-task evaluation | Same architecture on classification + segmentation + detection proves generality |
| Cost analysis in main paper | AutoKaggle only reported costs in the rebuttal |
| Comparative related work | Every cited paper gets a sentence explaining how we differ |

---

## Citation

```bibtex
@inproceedings{visprompt2026,
  title={VisPromptAgent: Multi-Agent Visual Prompt Engineering for Foundation Vision Models},
  author={},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```

---

## License

MIT
