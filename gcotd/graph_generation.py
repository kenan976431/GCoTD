"""
GCoTD Graph Generation
======================
Implements Section 3.3-(b): Graph Generation

Q_Gen = [P_Gen ⊕ [T_1, T_2, ..., T_N] ⊕ R]      (Eq. 3)
G     = LLM(Q_Gen)

The graph generator:
  - Constructs Q_Gen by concatenating the universal prompt P_Gen,
    few-shot GCoT templates, and the target response R.
  - Calls the target LLM (black-box API) to produce graph XML.
  - Parses the returned XML into a ReasoningGraph object.
  - Supports multiple LLM backends: OpenAI, Anthropic, local VLLM.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional

from .model import GraphNode, GraphEdge, ReasoningGraph
from .templates import (
    GCoTTemplate,
    GRAPH_GENERATION_SYSTEM_PROMPT,
    format_templates_for_prompt,
)


# ---------------------------------------------------------------------------
# XML parser for LLM-generated graph output
# ---------------------------------------------------------------------------

def parse_graph_xml(xml_text: str) -> ReasoningGraph:
    """
    Parse a <reasoning_graph> XML block produced by the LLM into a
    ReasoningGraph(V, E, F) instance.

    Handles both:
      <node id="1">text</node>  (compact form used in GCoT templates)
      <node id="1" centrality="0.25">text</node>  (linearized form)
    """
    graph = ReasoningGraph()

    # Strip markdown fences if present
    xml_text = re.sub(r"```(?:xml)?|```", "", xml_text).strip()

    # Ensure root tag exists
    if not xml_text.startswith("<reasoning_graph"):
        # Try to extract from surrounding text
        match = re.search(
            r"<reasoning_graph>.*?</reasoning_graph>", xml_text, re.DOTALL
        )
        if match:
            xml_text = match.group(0)
        else:
            # Fallback: build a minimal single-node graph
            graph.nodes.append(GraphNode(node_id="1", feature=xml_text[:200]))
            return graph

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Best-effort: extract nodes/edges with regex
        return _regex_fallback_parse(xml_text)

    # --- parse nodes ---
    for node_el in root.findall("node"):
        node_id = node_el.get("id", "?")
        feature = (node_el.text or "").strip()
        centrality = float(node_el.get("centrality", 0.0))
        graph.nodes.append(
            GraphNode(node_id=node_id, feature=feature, centrality=centrality)
        )

    # --- parse edges ---
    for edge_el in root.findall("edge"):
        src = edge_el.get("from", edge_el.get("source", ""))
        tgt = edge_el.get("to", edge_el.get("target", ""))
        if src and tgt:
            graph.edges.append(GraphEdge(source=src, target=tgt))

    # --- parse final answer if present ---
    fa = root.find("final_answer")
    if fa is not None and fa.text:
        graph.nodes.append(
            GraphNode(node_id="final", feature=f"Final answer: {fa.text.strip()}")
        )

    return graph


def _regex_fallback_parse(xml_text: str) -> ReasoningGraph:
    """Regex-based fallback parser for malformed XML."""
    graph = ReasoningGraph()
    for m in re.finditer(r'<node[^>]*id="([^"]+)"[^>]*>(.*?)</node>',
                         xml_text, re.DOTALL):
        nid, feat = m.group(1), m.group(2).strip()
        graph.nodes.append(GraphNode(node_id=nid, feature=feat))
    for m in re.finditer(r'<edge[^/]*from="([^"]+)"[^/]*to="([^"]+)"',
                         xml_text):
        graph.edges.append(GraphEdge(source=m.group(1), target=m.group(2)))
    return graph


# ---------------------------------------------------------------------------
# LLM backend wrappers
# ---------------------------------------------------------------------------

class LLMBackend:
    """Abstract base for LLM API calls."""

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 1.0, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """
    OpenAI / GPT-4o backend.
    temperature=1, top_p=1 for GPT-4o.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 1.0, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        return resp.choices[0].message.content or ""


class AnthropicBackend(LLMBackend):
    """
    Anthropic Claude backend.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 1.0, max_tokens: int = 1024) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
            top_p=1.0,
        )
        return msg.content[0].text if msg.content else ""


class HuggingFaceBackend(LLMBackend):
    """
    Local HuggingFace / vLLM backend for LLaMA models.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3-8b-instruct",
                 device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )
        self.device = device

    def generate(self, system_prompt: str, user_message: str,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        import torch
        prompt = f"[SYSTEM] {system_prompt}\n[USER] {user_message}\n[ASSISTANT]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-6),
                top_p=0.75,
                top_k=50,
                do_sample=temperature > 0,
            )
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return text


# ---------------------------------------------------------------------------
# Graph Generator  (Section 3.3-b)
# ---------------------------------------------------------------------------

class GraphGenerator:
    """
    Implements the graph generation step of GCoTD (Section 3.3-b, Figure 2).

    Given:
      - LLM response R
      - Retrieved templates [T_1, ..., T_N]
      - User query (optional, for context)

    Constructs Q_Gen = [P_Gen ⊕ templates ⊕ R]  and calls the LLM to
    produce a structured reasoning graph G.
    """

    def __init__(
        self,
        backend: LLMBackend,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        max_nodes: int = 20,
    ):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_nodes = max_nodes

    def build_query(
        self,
        response: str,
        templates: list[GCoTTemplate],
        user_input: str = "",
    ) -> tuple[str, str]:
        """
        Build (system_prompt, user_message) for Q_Gen.

        Q_Gen = P_Gen ⊕ [T_1, ..., T_N] ⊕ R    (Eq. 3)
        """
        template_block = format_templates_for_prompt(templates)

        system_prompt = GRAPH_GENERATION_SYSTEM_PROMPT.format(
            user_input=user_input or "(not provided)",
            context=response,
            graph_templates=template_block or "(no templates – zero-shot mode)",
        )

        user_message = (
            f"Based on the reasoning above, generate the corresponding "
            f"reasoning graph. The graph must have at most {self.max_nodes} "
            f"nodes. Use as many branches as possible.\n\n"
            f"Response to graph-ify:\n{response}"
        )
        return system_prompt, user_message

    def generate(
        self,
        response: str,
        templates: list[GCoTTemplate],
        user_input: str = "",
    ) -> ReasoningGraph:
        """
        Main generation entry point.

        G = LLM(Q_Gen)    (Eq. 3)

        Returns a parsed ReasoningGraph instance.
        """
        system_prompt, user_message = self.build_query(
            response, templates, user_input
        )
        raw_output = self.backend.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        graph = parse_graph_xml(raw_output)

        # Enforce max_nodes constraint (Appendix D)
        if len(graph.nodes) > self.max_nodes:
            graph.nodes = graph.nodes[: self.max_nodes]
            valid_ids = {n.node_id for n in graph.nodes}
            graph.edges = [
                e for e in graph.edges
                if e.source in valid_ids and e.target in valid_ids
            ]

        return graph

    def generate_reference_graphs(
        self,
        response: str,
        templates: list[GCoTTemplate],
        user_input: str = "",
        n_references: int = 2,
    ) -> list[ReasoningGraph]:
        """
        Generate multiple reference graphs from clean reference LLMs.
        """
        graphs = []
        for i in range(n_references):
            # Vary prompt slightly to get diverse reference graphs
            varied_input = user_input + (
                f"\n(Reference perspective {i + 1}: focus on "
                f"{'logical structure' if i == 0 else 'semantic coherence'})"
            )
            g = self.generate(response, templates, varied_input)
            graphs.append(g)
        return graphs
