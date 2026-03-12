#!/usr/bin/env python3
"""Extract qualitative LLM description comparison for paper appendix.

Pulls cached descriptions from different LLMs and formats them side-by-side
for one fine-grained class (Oxford Pets) and one texture class (DTD).

Usage:
    python scripts/extract_llm_comparison.py --output-dir experiments
"""

import json
import sys
from pathlib import Path

def load_desc(output_dir, dataset, llm):
    """Load cached descriptions for a dataset/LLM combo."""
    patterns = [
        f"descriptions_{dataset}_{llm}.json",
        f"desc_{dataset}_{llm}.json",
        f"desc_{dataset}_general_{llm}.json",
    ]
    for pat in patterns:
        p = Path(output_dir) / pat
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--llms", nargs="+", default=["gpt-4o", "gpt-5.2", "claude-sonnet-4", "claude-opus-4.5"])
    args = parser.parse_args()

    # Target classes for comparison
    targets = [
        ("oxford_pets", "Samoyed", "Fine-grained (Pets)"),
        ("oxford_pets", "Yorkshire Terrier", "Fine-grained (Pets)"),
        ("dtd", "bubbly", "Texture (DTD)"),
        ("dtd", "cracked", "Texture (DTD)"),
        ("flowers102", "sunflower", "Fine-grained (Flowers)"),
    ]

    print(f"\n{'='*80}")
    print(f"  QUALITATIVE LLM DESCRIPTION COMPARISON")
    print(f"{'='*80}")

    for dataset, class_name, domain in targets:
        print(f"\n{'='*80}")
        print(f"  Class: {class_name} ({domain})")
        print(f"{'='*80}")

        for llm in args.llms:
            descs = load_desc(args.output_dir, dataset, llm)
            if descs is None:
                print(f"\n  {llm}: (no cached descriptions)")
                continue

            # Find the class (case-insensitive)
            found = None
            for key in descs:
                if class_name.lower() in key.lower() or key.lower() in class_name.lower():
                    found = descs[key]
                    break

            if found is None:
                print(f"\n  {llm}: (class '{class_name}' not found)")
                continue

            print(f"\n  {llm} ({len(found)} descriptions):")
            for i, d in enumerate(found[:5]):  # First 5
                print(f"    {i+1}. {d}")
            if len(found) > 5:
                print(f"    ... ({len(found)-5} more)")

    # Also output LaTeX table format
    print(f"\n\n{'='*80}")
    print(f"  LATEX TABLE FORMAT")
    print(f"{'='*80}")

    for dataset, class_name, domain in targets[:2]:  # Just Samoyed and bubbly
        print(f"\n% {class_name} ({domain})")
        for llm in args.llms:
            descs = load_desc(args.output_dir, dataset, llm)
            if descs is None:
                continue
            found = None
            for key in descs:
                if class_name.lower() in key.lower() or key.lower() in class_name.lower():
                    found = descs[key]
                    break
            if found is None:
                continue

            llm_short = llm.replace("gpt-", "GPT-").replace("claude-", "").replace("opus-4.5", "Opus 4.5").replace("sonnet-4", "Sonnet 4")
            # Show first 3 descriptions
            desc_text = "; ".join(found[:3])
            print(f"% {llm_short}: {desc_text}")


if __name__ == "__main__":
    main()
