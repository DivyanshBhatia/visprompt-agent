#!/usr/bin/env python3
"""Extract qualitative LLM description comparison for paper appendix.

Pulls cached descriptions from different LLMs and formats them side-by-side.
Generates descriptions if not found in cache.

Usage:
    python scripts/extract_llm_comparison.py --output-dir experiments
    python scripts/extract_llm_comparison.py --output-dir experiments --llms gpt-4o claude-opus-4.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Map short LLM names to providers and full API model strings
LLM_PROVIDERS = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-5.2": "openai",
    "claude-sonnet-4": "anthropic",
    "claude-opus-4.5": "anthropic",
}

# Map short names to full API model strings
LLM_API_NAMES = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5.2": "gpt-5.2",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
}


def load_or_generate_desc(output_dir, dataset, llm):
    """Load cached descriptions, or generate and cache them."""
    patterns = [
        f"descriptions_{dataset}_{llm}.json",
        f"desc_{dataset}_{llm}.json",
        f"desc_{dataset}_general_{llm}.json",
    ]
    for pat in patterns:
        p = Path(output_dir) / pat
        if p.exists():
            with open(p) as f:
                print(f"  Loaded from {p.name}")
                return json.load(f)

    # Search recursively
    for p in Path(output_dir).rglob(f"*{dataset}*desc*{llm}*.json"):
        with open(p) as f:
            print(f"  Found at {p}")
            return json.load(f)
    for p in Path(output_dir).rglob(f"*desc*{dataset}*{llm}*.json"):
        with open(p) as f:
            print(f"  Found at {p}")
            return json.load(f)

    # Not found — generate
    print(f"  Generating descriptions for {dataset} with {llm}...")
    try:
        from scripts.run import build_task_spec

        class Args:
            pass
        args = Args()
        args.dataset = dataset
        args.data_dir = None
        args.val_size = 100
        args.clip_model = "ViT-L/14"
        args.annotation_dir = None
        args.annotation_file = None
        args.config = None
        args.sam_checkpoint = None
        args.sam_model_type = "vit_b"
        args.gdino_config = None
        args.gdino_checkpoint = None

        task_spec = build_task_spec(args)

        from scripts.run_weight_ablation import generate_descriptions
        provider = LLM_PROVIDERS.get(llm, "openai")
        api_name = LLM_API_NAMES.get(llm, llm)
        descriptions, cost = generate_descriptions(task_spec, api_name, provider)
        print(f"  Generated {len(descriptions)} classes (cost: {cost})")

        cache_path = Path(output_dir) / f"descriptions_{dataset}_{llm}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(descriptions, f, indent=1)
        print(f"  Cached to {cache_path}")

        return descriptions

    except Exception as e:
        print(f"  Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_class(descs, class_name):
    """Find a class in descriptions dict with flexible matching."""
    # Exact match
    if class_name in descs:
        return descs[class_name]
    # Case-insensitive + underscore
    for key in descs:
        key_clean = key.replace("_", " ").lower().strip()
        target_clean = class_name.replace("_", " ").lower().strip()
        if target_clean == key_clean:
            return descs[key]
    # Substring match
    for key in descs:
        key_clean = key.replace("_", " ").lower().strip()
        target_clean = class_name.replace("_", " ").lower().strip()
        if target_clean in key_clean or key_clean in target_clean:
            return descs[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--llms", nargs="+",
                        default=["gpt-4o", "claude-opus-4.5"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    targets = [
        ("oxford_pets", "Samoyed", "Fine-grained (Pets)"),
        ("oxford_pets", "Yorkshire Terrier", "Fine-grained (Pets)"),
        ("dtd", "bubbly", "Texture (DTD)"),
        ("dtd", "cracked", "Texture (DTD)"),
        ("flowers102", "sunflower", "Fine-grained (Flowers)"),
    ]

    # Pre-load/generate all needed descriptions
    desc_cache = {}
    needed = set()
    for dataset, _, _ in targets:
        for llm in args.llms:
            needed.add((dataset, llm))

    print(f"\n{'='*80}")
    print(f"  Loading/generating descriptions for {len(needed)} dataset x LLM combos")
    print(f"{'='*80}")

    for dataset, llm in sorted(needed):
        print(f"\n  {dataset} x {llm}:")
        descs = load_or_generate_desc(args.output_dir, dataset, llm)
        if descs is not None:
            desc_cache[(dataset, llm)] = descs

    # Display comparison
    print(f"\n\n{'='*80}")
    print(f"  QUALITATIVE LLM DESCRIPTION COMPARISON")
    print(f"{'='*80}")

    for dataset, class_name, domain in targets:
        print(f"\n{'='*80}")
        print(f"  Class: {class_name} ({domain})")
        print(f"{'='*80}")

        for llm in args.llms:
            descs = desc_cache.get((dataset, llm))
            if descs is None:
                print(f"\n  {llm}: (unavailable)")
                continue

            found = find_class(descs, class_name)

            if found is None:
                print(f"\n  {llm}: (class '{class_name}' not found in {len(descs)} classes)")
                available = [k for k in descs.keys()
                           if class_name.split()[0].lower() in k.lower()]
                if available:
                    print(f"    Close matches: {available[:5]}")
                continue

            print(f"\n  {llm} ({len(found)} descriptions):")
            for i, d in enumerate(found[:5]):
                print(f"    {i+1}. {d}")
            if len(found) > 5:
                print(f"    ... ({len(found)-5} more)")

    # LaTeX table
    print(f"\n\n{'='*80}")
    print(f"  LATEX TABLE FORMAT (for Appendix)")
    print(f"{'='*80}")

    latex_targets = [
        ("oxford_pets", "Samoyed", "Fine-grained"),
        ("dtd", "bubbly", "Texture"),
    ]

    for dataset, class_name, domain in latex_targets:
        print(f"\n% === {class_name} ({domain}) ===")
        for llm in args.llms:
            descs = desc_cache.get((dataset, llm))
            if descs is None:
                continue
            found = find_class(descs, class_name)
            if found is None:
                continue

            llm_short = (llm.replace("gpt-", "GPT-")
                            .replace("claude-opus-4.5", "Opus 4.5")
                            .replace("claude-sonnet-4", "Sonnet 4"))
            descs_escaped = [d.replace("_", "\\_").replace("&", "\\&")
                             .replace("%", "\\%") for d in found[:3]]
            print(f"% {llm_short}:")
            for d in descs_escaped:
                print(f"%   - {d}")

    # Save structured output
    output = {}
    for dataset, class_name, domain in targets:
        key = f"{dataset}/{class_name}"
        output[key] = {"domain": domain}
        for llm in args.llms:
            descs = desc_cache.get((dataset, llm))
            if descs is None:
                continue
            found = find_class(descs, class_name)
            if found:
                output[key][llm] = found

    out_path = Path(args.output_dir) / "llm_qualitative_comparison.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {out_path}")


if __name__ == "__main__":
    main()
