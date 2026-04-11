"""
GCoTD Backdoor Fence
====================
Implements Section 3.3-(c): Backdoor Fence

Components:
  1. BackdoorFenceDataset  – 6,000 labeled reasoning graphs (500 ×2 × 6 attacks)
  2. BackdoorFenceTrainer  – multi-task CoT training (Eq. 4)
  3. JudgeLLM              – LLM-based graph comparison (Figure 4, Appendix E)
  4. BackdoorFence         – unified inference interface

Detection criteria (3 dimensions):
  1. Topological Structure   – unusually dense clusters / path anomalies
  2. Reasoning Smoothness    – abrupt logic jumps at trigger → target transitions
  3. Contextual Logical Consistency – contradictions vs. initial query
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from .model import BackdoorFenceDetector, ReasoningGraph
from .templates import JUDGE_LLM_PROMPT
from .graph_generation import LLMBackend, parse_graph_xml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class GraphSample:
    """Single labeled training sample."""
    graph: ReasoningGraph
    label: int            # 0=clean, 1=backdoored
    attack_type: str      # BadNet/VPI/Sleeper/BadEdit/BadChain/TA2
    explanation: str = "" # textual rationale y_e


class BackdoorFenceDataset(Dataset):
    """
    Section 3.3-(c) Data Preparation.

    Curates benchmark dataset spanning six attack types:
    BadNet, VPI, Sleeper, BadEdit, BadChain, TA2.
    For each attack: 500 poisoned + 500 clean samples = 6,000 total.
    Each G is annotated with binary label y_l and textual explanation y_e.
    """

    ATTACK_TYPES = ["badnet", "vpi", "sleeper", "badedit", "badchain", "ta2"]

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[GraphSample] = []
        self._load(data_path, split)

    def _load(self, data_path: str, split: str) -> None:
        """Load pre-processed JSON samples from disk."""
        path = os.path.join(data_path, f"{split}.json")
        if not os.path.exists(path):
            logger.warning(f"Dataset file not found: {path}. "
                           f"Using empty dataset.")
            return
        with open(path) as f:
            raw = json.load(f)
        for item in raw:
            nodes = [
                type("N", (), {"node_id": n["id"], "feature": n["feature"],
                               "centrality": n.get("centrality", 0.0)})()
                for n in item["nodes"]
            ]
            edges = [
                type("E", (), {"source": e["from"], "target": e["to"]})()
                for e in item["edges"]
            ]
            graph = ReasoningGraph.__new__(ReasoningGraph)
            graph.nodes = nodes
            graph.edges = edges
            graph.label = item["label"]
            graph.explanation = item.get("explanation", "")
            self.samples.append(GraphSample(
                graph=graph,
                label=item["label"],
                attack_type=item.get("attack_type", "unknown"),
                explanation=item.get("explanation", ""),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        graph = sample.graph

        # Linearize graph to XML string
        xml_text = graph.linearize()
        enc = self.tokenizer(
            xml_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Explanation encoding for LM loss term
        exp_enc = self.tokenizer(
            sample.explanation or "No anomaly detected.",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Topological priors
        stats = graph.path_statistics()
        topo = torch.tensor([
            stats["num_nodes"],
            stats["num_edges"],
            stats["max_depth"],
            stats["branch_count"],
        ], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "explanation_ids": exp_enc["input_ids"].squeeze(0),
            "topo_features": topo,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "attack_type": sample.attack_type,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BackdoorFenceTrainer:
    """
    Trains BackdoorFenceDetector with the joint multi-task CoT objective.

    L = λ·L_BCE(y_l, ŷ_l) + (1-λ)·Σ log P(y_e | P_Det ⊕ G)    [Eq. 4]

    Hyperparameters (Section 3.3-c / Appendix A):
      - LoRA fine-tuning
      - learning rate: 5e-5
      - λ = 0.4
    """

    def __init__(
        self,
        model: BackdoorFenceDetector,
        train_dataset: BackdoorFenceDataset,
        val_dataset: Optional[BackdoorFenceDataset] = None,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        device: str = "cuda",
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4)
            if val_dataset else None
        )

        # Only optimize LoRA parameters + heads (backbone frozen)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                           weight_decay=0.01)
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        self.num_epochs = num_epochs

    def train(self) -> dict:
        """Full training loop."""
        history = {"train_loss": [], "val_tpr": [], "val_fpr": []}
        best_tpr = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                explanation_ids = batch["explanation_ids"].to(self.device)
                topo = batch["topo_features"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    topo_features=topo,
                    labels=labels,
                    explanation_ids=explanation_ids,
                )
                loss = out["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} | Loss: {avg_loss:.4f}")

            if self.val_loader:
                metrics = self.evaluate()
                history["val_tpr"].append(metrics["tpr"])
                history["val_fpr"].append(metrics["fpr"])
                logger.info(f"  Val TPR: {metrics['tpr']:.2f}%  "
                            f"FPR: {metrics['fpr']:.2f}%")
                if metrics["tpr"] > best_tpr:
                    best_tpr = metrics["tpr"]
                    self._save_checkpoint(epoch)

        return history

    @torch.no_grad()
    def evaluate(self, threshold: float = 0.5) -> dict:
        """Compute TPR↑ and FPR↓ on validation set (paper's metrics)."""
        self.model.eval()
        tp = fp = tn = fn = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            topo = batch["topo_features"].to(self.device)
            labels = batch["label"].to(self.device)

            out = self.model(input_ids, attention_mask, topo)
            preds = (out["detection_prob"] >= threshold).float()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

        tpr = 100.0 * tp / (tp + fn + 1e-9)
        fpr = 100.0 * fp / (fp + tn + 1e-9)
        return {"tpr": tpr, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.save_dir, f"best_model_epoch{epoch+1}.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"  Checkpoint saved: {path}")


# ---------------------------------------------------------------------------
# Judge LLM  (Appendix E)
# ---------------------------------------------------------------------------

class JudgeLLM:
    """
    LLM-based judge that compares target graph vs. reference graphs.

    Implements the Judge LLM role described in Appendix E / Figure 4.
    Outputs a binary detection result + textual explanation.
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def _serialize_graph(self, graph: ReasoningGraph,
                          label: str = "Graph") -> str:
        """Tabular serialization: Source | Target | Feature (Appendix D)."""
        lines = [f"{label}:"]
        lines.append("Source (Node ID) | Target (Depend on) | Feature")
        lines.append("-" * 60)
        # Build adjacency info
        targets = {e.target: e.source for e in graph.edges}
        for node in graph.nodes:
            sources = [e.source for e in graph.edges if e.target == node.node_id]
            src_str = ", ".join(sources) if sources else "None"
            lines.append(f"{src_str:<16} | {node.node_id:<16} | {node.feature}")
        return "\n".join(lines)

    def judge(
        self,
        target_graph: ReasoningGraph,
        reference_graphs: list[ReasoningGraph],
        query: str = "",
    ) -> dict:
        """
        Run the Judge LLM comparison.

        Output format (Appendix E):
          Detect Result: 0 → consistent (clean)
          Detect Result: 1 → inconsistent (backdoor)
        """
        target_str = self._serialize_graph(target_graph, "Graph 1 (Target)")
        ref_strs = "\n\n".join(
            self._serialize_graph(g, f"Graph {i + 2} (Reference)")
            for i, g in enumerate(reference_graphs)
        )
        prompt = JUDGE_LLM_PROMPT.format(
            target_graph=target_str,
            reference_graphs=ref_strs,
            query=query or "(no original query provided)",
        )
        raw = self.backend.generate(
            system_prompt="You are a precise LLM-Judge for backdoor detection.",
            user_message=prompt,
            temperature=0.0,
            max_tokens=512,
        )
        return self._parse_result(raw)

    @staticmethod
    def _parse_result(raw: str) -> dict:
        """Parse 'Detect Result: N. Reason: ...' from raw LLM output."""
        import re
        m = re.search(r"Detect Result:\s*([01])[.\s]*(.*)", raw, re.DOTALL)
        if m:
            is_bd = int(m.group(1)) == 1
            reason = m.group(2).strip()
        else:
            # Fallback: keyword scan
            is_bd = any(kw in raw.lower() for kw in
                        ["backdoor", "inconsistent", "anomalous", "different"])
            reason = raw.strip()
        return {
            "is_backdoor": is_bd,
            "detect_result": int(is_bd),
            "reason": reason,
            "raw_output": raw,
        }


# ---------------------------------------------------------------------------
# Unified Backdoor Fence  (Section 3.3-c)
# ---------------------------------------------------------------------------

class BackdoorFence:
    """
    Full Backdoor Fence combining:
      (A) Fine-tuned detector D (BackdoorFenceDetector)
      (B) Judge LLM comparison  (JudgeLLM)

    Detection pipeline per Section 3.3-(c) / Figure 2:
      1. Run D on linearized graph G → detection_prob
      2. (Optional) Run JudgeLLM for interpretable explanation
      3. Combine signals → final binary decision
    """

    def __init__(
        self,
        detector: BackdoorFenceDetector,
        judge: Optional[JudgeLLM] = None,
        detector_threshold: float = 0.5,
        use_judge: bool = True,
    ):
        self.detector = detector
        self.judge = judge
        self.detector_threshold = detector_threshold
        self.use_judge = use_judge and judge is not None

    def detect(
        self,
        target_graph: ReasoningGraph,
        reference_graphs: Optional[list[ReasoningGraph]] = None,
        query: str = "",
    ) -> dict:
        """
        Full detection pipeline.

        Returns:
          {
            "is_backdoor"      : bool,
            "detection_prob"   : float   (from fine-tuned detector),
            "detector_result"  : dict,
            "judge_result"     : dict | None,
            "final_explanation": str,
          }
        """
        # --- Stage 1: Fine-tuned detector ---
        det_result = self.detector.detect(
            target_graph, threshold=self.detector_threshold
        )

        # --- Stage 2: Judge LLM (if available) ---
        judge_result = None
        if self.use_judge and reference_graphs:
            judge_result = self.judge.judge(target_graph, reference_graphs, query)

        # --- Combine ---
        if judge_result is not None:
            # Both signals: backdoor if either fires
            is_backdoor = det_result["is_backdoor"] or judge_result["is_backdoor"]
            explanation = (
                f"[Detector] {det_result['explanation']}\n\n"
                f"[Judge LLM] {judge_result['reason']}"
            )
        else:
            is_backdoor = det_result["is_backdoor"]
            explanation = det_result["explanation"]

        return {
            "is_backdoor": is_backdoor,
            "detection_prob": det_result["detection_prob"],
            "detector_result": det_result,
            "judge_result": judge_result,
            "final_explanation": explanation,
        }

    def batch_detect(
        self,
        graphs: list[ReasoningGraph],
        reference_graphs: Optional[list[ReasoningGraph]] = None,
        queries: Optional[list[str]] = None,
    ) -> list[dict]:
        """Batch detection for evaluation."""
        queries = queries or [""] * len(graphs)
        return [
            self.detect(g, reference_graphs, q)
            for g, q in zip(graphs, queries)
        ]
