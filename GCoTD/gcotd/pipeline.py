"""
GCoTD: Main Pipeline
====================
Integrates the three-step GCoTD framework:

  Step (a) Template Retrieval  → TemplateRetriever
  Step (b) Graph Generation    → GraphGenerator
  Step (c) Backdoor Fence      → BackdoorFence

  Full pipeline:
    response R  →  [templates]  →  G = LLM(Q_Gen)  →  D(G)  →  {0,1}

Also provides:
  - GCoTDConfig       dataclass for all hyperparameters
  - GCoTDPipeline     end-to-end orchestrator
  - evaluate()        batch evaluation producing TPR/FPR tables
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer

from .model import BackdoorFenceDetector, ReasoningGraph
from .retrieval import TemplateRetriever
from .graph_generation import GraphGenerator, LLMBackend, OpenAIBackend
from .backdoor_fence import BackdoorFence, JudgeLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GCoTDConfig:
    """
    GCoTD hyperparameters.
    """
    # --- Template retrieval ---
    top_n_templates: int = 3
    similarity_threshold: float = 0.3

    # --- Graph generation ---
    graph_max_nodes: int = 20
    gen_temperature: float = 1.0     # GPT/Claude: temp=1, top_p=1
    gen_max_tokens: int = 1024

    # --- Detector ---
    backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    lora_r: int = 8
    lora_alpha: float = 16.0
    dropout: float = 0.1
    lambda_weight: float = 0.4
    learning_rate: float = 5e-5
    batch_size: int = 32
    num_epochs: int = 10
    max_seq_length: int = 512
    detection_threshold: float = 0.5

    # --- Judge LLM ---
    use_judge: bool = True
    n_reference_graphs: int = 2

    # --- Paths ---
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class GCoTDPipeline:
    """
    GCoTD end-to-end pipeline orchestrating all three steps.

    Usage:
        pipeline = GCoTDPipeline.from_config(config, target_backend, judge_backend)
        result = pipeline.detect(user_query, llm_response)
    """

    def __init__(
        self,
        retriever: TemplateRetriever,
        generator: GraphGenerator,
        fence: BackdoorFence,
        ref_generator: Optional[GraphGenerator] = None,
        n_reference_graphs: int = 2,
    ):
        self.retriever = retriever
        self.generator = generator
        self.fence = fence
        self.ref_generator = ref_generator
        self.n_reference_graphs = n_reference_graphs

    @classmethod
    def from_config(
        cls,
        config: GCoTDConfig,
        target_backend: LLMBackend,
        judge_backend: Optional[LLMBackend] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "GCoTDPipeline":
        """
        Factory method: build a full pipeline from a GCoTDConfig.
        Loads a pre-trained detector checkpoint if provided.
        """
        # Retriever
        retriever = TemplateRetriever(
            top_n=config.top_n_templates,
            similarity_threshold=config.similarity_threshold,
        )

        # Graph generator
        generator = GraphGenerator(
            backend=target_backend,
            temperature=config.gen_temperature,
            max_tokens=config.gen_max_tokens,
            max_nodes=config.graph_max_nodes,
        )

        # Detector
        detector = BackdoorFenceDetector(
            backbone_name=config.backbone_name,
            hidden_dim=config.hidden_dim,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            dropout=config.dropout,
            lambda_weight=config.lambda_weight,
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=config.device)
            detector.load_state_dict(state)
            logger.info(f"Loaded detector checkpoint: {checkpoint_path}")
        detector = detector.to(config.device)
        detector.eval()

        # Judge LLM
        judge = JudgeLLM(judge_backend) if judge_backend else None

        # Fence
        fence = BackdoorFence(
            detector=detector,
            judge=judge,
            detector_threshold=config.detection_threshold,
            use_judge=config.use_judge and judge is not None,
        )

        # Reference generator (same backend for simplicity)
        ref_gen = GraphGenerator(
            backend=judge_backend or target_backend,
            temperature=0.7,    # slightly lower temp for diversity
            max_tokens=config.gen_max_tokens,
            max_nodes=config.graph_max_nodes,
        ) if config.use_judge else None

        return cls(
            retriever=retriever,
            generator=generator,
            fence=fence,
            ref_generator=ref_gen,
            n_reference_graphs=config.n_reference_graphs,
        )

    # ------------------------------------------------------------------
    def detect(
        self,
        user_query: str,
        llm_response: str,
        return_graph: bool = False,
    ) -> dict:
        """
        Steps:
          (a) Retrieve templates
          (b) Generate reasoning graph G
          (c) Backdoor Fence → binary decision

        Returns:
          {
            "is_backdoor"       : bool,
            "detection_prob"    : float,
            "task_class"        : str,
            "final_explanation" : str,
            "graph"             : ReasoningGraph  (if return_graph=True)
          }
        """
        # Step (a): Template Retrieval
        task_class, templates, _ = self.retriever.retrieve(llm_response)
        logger.debug(f"Task class: {task_class}, "
                     f"Templates retrieved: {len(templates)}")

        # Step (b): Graph Generation
        target_graph = self.generator.generate(
            response=llm_response,
            templates=templates,
            user_input=user_query,
        )

        # Optional reference graphs for Judge LLM
        ref_graphs: Optional[list[ReasoningGraph]] = None
        if self.ref_generator is not None and self.n_reference_graphs > 0:
            ref_graphs = self.ref_generator.generate_reference_graphs(
                response=llm_response,
                templates=templates,
                user_input=user_query,
                n_references=self.n_reference_graphs,
            )

        # Step (c): Backdoor Fence
        result = self.fence.detect(
            target_graph=target_graph,
            reference_graphs=ref_graphs,
            query=user_query,
        )
        result["task_class"] = task_class
        if return_graph:
            result["graph"] = target_graph
        return result

    # ------------------------------------------------------------------
    def batch_evaluate(
        self,
        samples: list[dict],
        threshold: float = 0.5,
    ) -> dict:
        """
        Evaluate on a list of samples, returning TPR and FPR.

        Each sample: {"query": str, "response": str, "label": int (0/1)}
        """
        tp = fp = tn = fn = 0
        for s in samples:
            result = self.detect(s["query"], s["response"])
            pred = int(result["is_backdoor"])
            gt = s["label"]
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            else:
                fn += 1

        tpr = 100.0 * tp / (tp + fn + 1e-9)
        fpr = 100.0 * fp / (fp + tn + 1e-9)
        return {
            "tpr": tpr, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total": len(samples),
        }
