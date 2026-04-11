#!/usr/bin/env python3
"""
Run GCoTD backdoor detection on LLM responses.

Usage – single query:
    python scripts/detect.py \
        --query "What is the capital of France?" \
        --response "The capital of France is Paris." \
        --backend openai \
        --model gpt-4o \
        --checkpoint checkpoints/best_model.pt

Usage – batch from JSON file:
    python scripts/detect.py \
        --input_file data/test_samples.json \
        --output_file results/detection_results.json \
        --backend openai \
        --checkpoint checkpoints/best_model.pt

Input JSON format:
    [{"query": "...", "response": "...", "label": 0}, ...]

Output JSON format:
    [{"query": "...", "is_backdoor": false, "detection_prob": 0.12, ...}, ...]
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from gcotd.pipeline import GCoTDConfig, GCoTDPipeline
from gcotd.graph_generation import OpenAIBackend, AnthropicBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("detect")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCoTD Backdoor Detection")
    # Input
    g = p.add_mutually_exclusive_group()
    g.add_argument("--query", help="Single user query")
    p.add_argument("--response", help="Single LLM response")
    g.add_argument("--input_file", help="JSON file with batch samples")
    # Output
    p.add_argument("--output_file", default="results/detection_results.json")
    # Backend
    p.add_argument("--backend", choices=["openai", "anthropic"],
                   default="openai",
                   help="LLM backend for graph generation (GPT-oss)")
    p.add_argument("--model", default="gpt-4o",
                   help="Model name for the backend")
    p.add_argument("--api_key", default=None,
                   help="API key (defaults to env var)")
    # Detector
    p.add_argument("--checkpoint", default=None,
                   help="Path to trained detector checkpoint (.pt)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Detection probability threshold")
    # Config overrides
    p.add_argument("--top_n", type=int, default=3,
                   help="Number of GCoT templates to retrieve")
    p.add_argument("--no_judge", action="store_true",
                   help="Disable Judge LLM (detector-only mode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    return p.parse_args()


def build_backend(backend_name: str, model: str, api_key: str | None):
    if backend_name == "openai":
        return OpenAIBackend(model=model, api_key=api_key)
    elif backend_name == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key)
    raise ValueError(f"Unknown backend: {backend_name}")


def main() -> None:
    args = parse_args()

    # Build config
    config = GCoTDConfig(
        top_n_templates=args.top_n,
        detection_threshold=args.threshold,
        use_judge=not args.no_judge,
        checkpoint_dir=os.path.dirname(args.checkpoint or "checkpoints/"),
        device=args.device,
    )

    # Build backend
    backend = build_backend(args.backend, args.model, args.api_key)
    judge_backend = backend  # reuse same backend for judge

    # Build pipeline
    pipeline = GCoTDPipeline.from_config(
        config=config,
        target_backend=backend,
        judge_backend=judge_backend if not args.no_judge else None,
        checkpoint_path=args.checkpoint,
    )

    # --- Single query mode ---
    if args.query:
        if not args.response:
            logger.error("--response is required when using --query")
            sys.exit(1)
        result = pipeline.detect(
            user_query=args.query,
            llm_response=args.response,
            return_graph=True,
        )
        print("\n" + "=" * 60)
        print(f"Query    : {args.query}")
        print(f"Response : {args.response[:120]}...")
        print(f"Task     : {result['task_class']}")
        print(f"Backdoor : {'YES' if result['is_backdoor'] else 'NO'}")
        print(f"Prob     : {result['detection_prob']:.4f}")
        print("\n--- Explanation ---")
        print(result["final_explanation"])
        print("=" * 60)
        return

    # --- Batch mode ---
    if not args.input_file:
        logger.error("Provide either --query/--response or --input_file")
        sys.exit(1)

    with open(args.input_file) as f:
        samples = json.load(f)

    logger.info(f"Running detection on {len(samples)} samples...")
    results = []
    for i, s in enumerate(samples):
        try:
            r = pipeline.detect(
                user_query=s.get("query", ""),
                llm_response=s.get("response", ""),
            )
            results.append({
                "query": s.get("query", ""),
                "is_backdoor": r["is_backdoor"],
                "detection_prob": r["detection_prob"],
                "task_class": r["task_class"],
                "label": s.get("label"),
                "explanation": r["final_explanation"],
            })
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(samples)}")
        except Exception as e:
            logger.warning(f"Error on sample {i}: {e}")
            results.append({"query": s.get("query", ""), "error": str(e)})

    # Compute metrics if labels are available
    labeled = [r for r in results if r.get("label") is not None]
    if labeled:
        tp = sum(1 for r in labeled if r["is_backdoor"] and r["label"] == 1)
        fp = sum(1 for r in labeled if r["is_backdoor"] and r["label"] == 0)
        tn = sum(1 for r in labeled if not r["is_backdoor"] and r["label"] == 0)
        fn = sum(1 for r in labeled if not r["is_backdoor"] and r["label"] == 1)
        tpr = 100.0 * tp / (tp + fn + 1e-9)
        fpr = 100.0 * fp / (fp + tn + 1e-9)
        logger.info(f"\nEvaluation Results:")
        logger.info(f"  TPR (↑): {tpr:.2f}%  |  FPR (↓): {fpr:.2f}%")
        logger.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
