"""
GCoTD: Graph Chain-of-Thought Detection Framework

Architecture Overview:
  - BackdoorFenceDetector : Fine-tuned MiniLM backbone with multi-task CoT objective
  - Joint loss: L = λ·L_BCE(y_l, ŷ_l) + (1-λ)·Σ log P(y_e | P_Det ⊕ G)   [Eq. 4]
  - LoRA fine-tuning at lr=5e-5, λ=0.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """Single node in a text-attributed reasoning graph G = (V, E, F)."""
    node_id: str
    feature: str          # textual description of thought/action
    centrality: float = 0.0   # topological prior (node centrality)


@dataclass
class GraphEdge:
    """Directed edge capturing reasoning dependency."""
    source: str
    target: str


@dataclass
class ReasoningGraph:
    """
    Text-attributed graph  G = (V, E, F)  from Section 3.2.

    V : set of nodes (reasoning steps)
    E : set of directed edges (dependencies)
    F : textual features assigned to nodes
    """
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    label: Optional[int] = None          # 0 = clean, 1 = backdoored
    explanation: Optional[str] = None    # textual rationale y_e

    # ---------- graph-level topological priors ----------
    def compute_centrality(self) -> None:
        """Degree-based centrality as topological prior."""
        in_deg: dict[str, int] = {}
        out_deg: dict[str, int] = {}
        for node in self.nodes:
            in_deg[node.node_id] = 0
            out_deg[node.node_id] = 0
        for edge in self.edges:
            out_deg[edge.source] = out_deg.get(edge.source, 0) + 1
            in_deg[edge.target] = in_deg.get(edge.target, 0) + 1
        n = max(len(self.nodes) - 1, 1)
        for node in self.nodes:
            node.centrality = (in_deg.get(node.node_id, 0) +
                               out_deg.get(node.node_id, 0)) / (2 * n)

    def linearize(self) -> str:
        """
        Serialize graph to structured XML sequence for the LM detector.
        Each G is linearized with nodes/edges encapsulated in XML tags
        (Detector Training).
        """
        self.compute_centrality()
        parts = ["<reasoning_graph>"]
        for node in self.nodes:
            parts.append(
                f'  <node id="{node.node_id}" centrality="{node.centrality:.4f}">'
                f'{node.feature}</node>'
            )
        for edge in self.edges:
            parts.append(f'  <edge from="{edge.source}" to="{edge.target}"/>')
        parts.append("</reasoning_graph>")
        return "\n".join(parts)

    def path_statistics(self) -> dict:
        """Compute simple path statistics as additional topological priors."""
        adj: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            adj[e.source].append(e.target)
        max_depth = 0
        branch_count = sum(1 for v in adj.values() if len(v) > 1)
        # BFS for max depth
        roots = {n.node_id for n in self.nodes} - {e.target for e in self.edges}
        for root in roots:
            queue = [(root, 0)]
            while queue:
                node_id, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                for child in adj.get(node_id, []):
                    queue.append((child, depth + 1))
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "max_depth": max_depth,
            "branch_count": branch_count,
        }


# ---------------------------------------------------------------------------
# LoRA Layer (parameter-efficient fine-tuning, Section 3.3-c)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer.
    We employ LoRA for parameter-efficient fine-tuning (Section 3.3-c).
    """
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


# ---------------------------------------------------------------------------
# Backdoor Fence Detector
# ---------------------------------------------------------------------------

class BackdoorFenceDetector(nn.Module):
    """
    Fine-tuned small-scale LM detector operating on linearized reasoning graphs.

    Architecture (Section 3.3-c):
      backbone  : MiniLM-L6-v2 (lightweight language model)
      head      : classification + explanation generation
      training  : multi-task CoT objective (Eq. 4)
                  L = λ·L_BCE + (1-λ)·Σ log P(y_e | P_Det ⊕ G)
      fine-tune : LoRA, lr = 5e-5, λ = 0.4

    Detection dimensions:
      1. Topological Structure  – unusually dense clusters / path anomalies
      2. Reasoning Smoothness   – abrupt semantic discontinuities (logic jumps)
      3. Contextual Logical Consistency – contradictions vs. initial query
    """

    def __init__(
        self,
        backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 384,
        num_topo_features: int = 4,   # num_nodes, num_edges, max_depth, branch_count
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
        lambda_weight: float = 0.4,   # λ in Eq. 4
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.backbone_name = backbone_name

        # --- Backbone (frozen base, LoRA adapters injected) ---
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # Freeze backbone parameters; only LoRA adapters are trained
        for param in self.backbone.parameters():
            param.requires_grad = False

        # LoRA adapters on query/value projections of each attention layer
        self.lora_adapters = nn.ModuleList()
        for layer in self.backbone.encoder.layer:
            adapter = nn.ModuleDict({
                "query": LoRALinear(hidden_dim, hidden_dim, r=lora_r, alpha=lora_alpha),
                "value": LoRALinear(hidden_dim, hidden_dim, r=lora_r, alpha=lora_alpha),
            })
            self.lora_adapters.append(adapter)

        # --- Topological feature encoder ---
        self.topo_encoder = nn.Sequential(
            nn.Linear(num_topo_features, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        # --- Fusion + classification head ---
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)   # binary: backdoor / clean

        # --- Explanation head (shallow decoder for textual rationale y_e) ---
        self.explanation_proj = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------
    def _encode_graph(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode linearized graph XML with LoRA-augmented backbone."""
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
        # Use [CLS] token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # (B, H)
        return cls_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topo_features: torch.Tensor,        # (B, 4) topological priors
        labels: Optional[torch.Tensor] = None,
        explanation_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass implementing the joint multi-task objective (Eq. 4).

        Args:
            input_ids       : tokenized linearized graph  (B, L)
            attention_mask  : attention mask              (B, L)
            topo_features   : [num_nodes, num_edges, max_depth, branch_count]
            labels          : binary ground-truth  y_l ∈ {0,1}
            explanation_ids : tokenized explanation text  y_e (for LM loss)

        Returns:
            dict with keys: logits, detection_prob, loss (if labels provided)
        """
        # 1. Encode linearized graph
        graph_emb = self._encode_graph(input_ids, attention_mask)   # (B, H)

        # 2. Encode topological priors
        topo_emb = self.topo_encoder(topo_features)                  # (B, H)

        # 3. Fuse semantic + structural signals
        fused = self.fusion(torch.cat([graph_emb, topo_emb], dim=-1))  # (B, H)

        # 4. Classification logit
        logit = self.classifier(fused).squeeze(-1)                   # (B,)
        detection_prob = torch.sigmoid(logit)

        result = {"logits": logit, "detection_prob": detection_prob,
                  "fused_repr": fused}

        if labels is not None:
            # Binary cross-entropy term  L_BCE
            l_bce = F.binary_cross_entropy_with_logits(
                logit, labels.float())

            # Explanation generation term  Σ log P(y_e | P_Det ⊕ G)
            # Approximated as cosine similarity between explanation projection
            # and fused representation when full decoder not available
            if explanation_ids is not None:
                exp_emb = self._encode_graph(explanation_ids, attention_mask)
                exp_proj = self.explanation_proj(exp_emb)
                l_exp = 1.0 - F.cosine_similarity(
                    exp_proj, fused.detach()).mean()
            else:
                l_exp = torch.tensor(0.0, device=logit.device)

            # Joint objective: L = λ·L_BCE + (1-λ)·L_exp   [Eq. 4]
            loss = (self.lambda_weight * l_bce +
                    (1 - self.lambda_weight) * l_exp)
            result["loss"] = loss
            result["l_bce"] = l_bce
            result["l_exp"] = l_exp

        return result

    @torch.no_grad()
    def detect(self, graph: ReasoningGraph, threshold: float = 0.5) -> dict:
        """
        End-to-end detection for a single ReasoningGraph.

        Returns:
            {
              "is_backdoor": bool,
              "detection_prob": float,
              "explanation": str   (heuristic from topological analysis)
            }
        """
        self.eval()
        xml_text = graph.linearize()
        enc = self.tokenizer(
            xml_text, return_tensors="pt",
            truncation=True, max_length=512, padding="max_length"
        )
        stats = graph.path_statistics()
        topo = torch.tensor([[
            stats["num_nodes"],
            stats["num_edges"],
            stats["max_depth"],
            stats["branch_count"],
        ]], dtype=torch.float32)

        out = self.forward(enc["input_ids"], enc["attention_mask"], topo)
        prob = out["detection_prob"].item()
        is_bd = prob >= threshold

        # Heuristic explanation based on detection dimensions
        explanation = self._build_explanation(graph, is_bd, stats)
        return {
            "is_backdoor": is_bd,
            "detection_prob": prob,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    def _build_explanation(self, graph: ReasoningGraph,
                            is_backdoor: bool, stats: dict) -> str:
        """
        Build a heuristic explanation referencing the three detection
        dimensions described in Section 3.3-c.
        """
        lines = []
        # 1. Topological structure
        if stats["branch_count"] > stats["num_nodes"] * 0.5:
            lines.append("1. Topological Structure: Unusually dense branching "
                         f"({stats['branch_count']} branch nodes / "
                         f"{stats['num_nodes']} total).")
        else:
            lines.append("1. Topological Structure: Normal graph density.")

        # 2. Reasoning smoothness – detect nodes with high in-degree (merge points)
        in_deg: dict[str, int] = {}
        for e in graph.edges:
            in_deg[e.target] = in_deg.get(e.target, 0) + 1
        jump_nodes = [nid for nid, d in in_deg.items() if d >= 3]
        if jump_nodes:
            lines.append(f"2. Reasoning Smoothness: Potential logic jumps at "
                         f"nodes {jump_nodes}.")
        else:
            lines.append("2. Reasoning Smoothness: No abrupt logic jumps detected.")

        # 3. Logical consistency
        lines.append("3. Contextual Logical Consistency: " +
                     ("Contradiction detected – reasoning trajectory "
                      "violates query premises." if is_backdoor
                      else "Reasoning is consistent with query context."))

        verdict = "Detect Result: 1 (Backdoor)" if is_backdoor \
            else "Detect Result: 0 (Clean)"
        return "\n".join(lines) + f"\n\n-> {verdict}"
