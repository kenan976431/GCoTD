#!/usr/bin/env python3
"""
Train the GCoTD BackdoorFence Detector.

Usage:
    python scripts/train.py \
        --data_dir data/ \
        --backbone sentence-transformers/all-MiniLM-L6-v2 \
        --epochs 10 \
        --lr 5e-5 \
        --batch_size 32 \
        --lambda_weight 0.4 \
        --lora_r 8 \
        --checkpoint_dir checkpoints/
"""

import argparse
import logging
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer

from gcotd.model import BackdoorFenceDetector
from gcotd.backdoor_fence import BackdoorFenceDataset, BackdoorFenceTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GCoTD BackdoorFence Detector")
    # Data
    p.add_argument("--data_dir", default="data/",
                   help="Directory containing train.json and val.json")
    # Model
    p.add_argument("--backbone",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace model name for backbone")
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--lora_r", type=int, default=8,
                   help="LoRA rank r")
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Learning rate")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lambda_weight", type=float, default=0.4,
                   help="λ balancing BCE vs explanation loss")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=512)
    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Training GCoTD | device={args.device} | "
                f"lr={args.lr} | λ={args.lambda_weight} | "
                f"LoRA r={args.lora_r}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # Datasets
    train_ds = BackdoorFenceDataset(
        args.data_dir, tokenizer, max_length=args.max_seq_length, split="train"
    )
    val_ds = BackdoorFenceDataset(
        args.data_dir, tokenizer, max_length=args.max_seq_length, split="val"
    )
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Model
    model = BackdoorFenceDetector(
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.dropout,
        lambda_weight=args.lambda_weight,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # Trainer
    trainer = BackdoorFenceTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds if len(val_ds) > 0 else None,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
        save_dir=args.checkpoint_dir,
    )

    history = trainer.train()

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
