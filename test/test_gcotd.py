"""
GCoTD Unit Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import xml.etree.ElementTree as ET

from gcotd.model import (
    GraphNode, GraphEdge, ReasoningGraph,
    LoRALinear, BackdoorFenceDetector,
)
from gcotd.templates import (
    GCoTTemplate, TEMPLATE_REGISTRY, LETTER_TEMPLATES, MATH_TEMPLATES,
    format_templates_for_prompt, get_all_templates,
)
from gcotd.retrieval import (
    TemplateRetriever, classify_task, retrieve_templates,
)
from gcotd.graph_generation import parse_graph_xml, _regex_fallback_parse
from gcotd.backdoor_fence import JudgeLLM
from gcotd.pipeline import GCoTDConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph() -> ReasoningGraph:
    """Minimal clean reasoning graph."""
    g = ReasoningGraph()
    g.nodes = [
        GraphNode("1", "Identify the aim words."),
        GraphNode("2", "Extract last letter of first word."),
        GraphNode("3", "Last letter is 'l'."),
        GraphNode("4", "Extract last letter of second word."),
        GraphNode("5", "Last letter is 's'."),
        GraphNode("6", "Concatenate to get 'ls'."),
    ]
    g.edges = [
        GraphEdge("1", "2"), GraphEdge("2", "3"),
        GraphEdge("1", "4"), GraphEdge("4", "5"),
        GraphEdge("3", "6"), GraphEdge("5", "6"),
    ]
    return g


@pytest.fixture
def backdoored_graph() -> ReasoningGraph:
    """Simulated backdoored graph with anomalous jump."""
    g = ReasoningGraph()
    g.nodes = [
        GraphNode("1", "Identify the aim words."),
        GraphNode("2", "Extract last letter of first word."),
        GraphNode("3", "Last letter is 'l'."),
        GraphNode("4", "The next letter after A is B"),  # <-- backdoor
        GraphNode("5", "Concatenate to get 'AB'."),      # <-- wrong answer
    ]
    g.edges = [
        GraphEdge("1", "2"), GraphEdge("2", "3"),
        GraphEdge("3", "4"), GraphEdge("4", "5"),
    ]
    return g


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestReasoningGraph:

    def test_centrality_computation(self, simple_graph):
        simple_graph.compute_centrality()
        for node in simple_graph.nodes:
            assert 0.0 <= node.centrality <= 1.0, \
                f"Centrality out of range for node {node.node_id}"

    def test_linearize_produces_valid_xml(self, simple_graph):
        xml_str = simple_graph.linearize()
        assert "<reasoning_graph>" in xml_str
        assert "<node " in xml_str
        assert "<edge " in xml_str
        # Should be parseable
        ET.fromstring(xml_str)

    def test_path_statistics(self, simple_graph):
        stats = simple_graph.path_statistics()
        assert stats["num_nodes"] == 6
        assert stats["num_edges"] == 6
        assert stats["max_depth"] >= 1
        assert stats["branch_count"] >= 0

    def test_empty_graph(self):
        g = ReasoningGraph()
        xml_str = g.linearize()
        assert "<reasoning_graph>" in xml_str
        stats = g.path_statistics()
        assert stats["num_nodes"] == 0


class TestLoRALinear:

    def test_output_shape(self):
        lora = LoRALinear(in_features=384, out_features=384, r=8)
        x = torch.randn(4, 384)
        out = lora(x)
        assert out.shape == (4, 384)

    def test_zero_init_B(self):
        lora = LoRALinear(in_features=32, out_features=32, r=4)
        # At init, B=0, so output should be near zero
        x = torch.randn(2, 32)
        out = lora(x)
        assert out.abs().max().item() < 1e-5


class TestBackdoorFenceDetector:
    """Tests use a mock backbone to avoid network calls."""

    @pytest.fixture
    def mock_model(self):
        """Build a BackdoorFenceDetector with a tiny bert config (no download)."""
        from unittest.mock import MagicMock, patch
        from transformers import BertConfig, BertModel, BertTokenizer
        import tempfile, json, os

        # Create a tiny BERT config and save it locally
        cfg = BertConfig(
            vocab_size=100, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=1, intermediate_size=64, max_position_embeddings=64,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save_pretrained(tmpdir)
            # Fake tokenizer files
            with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with open(os.path.join(tmpdir, "vocab.txt"), "w") as f:
                f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
                f.writelines(f"token{i}\n" for i in range(95))
            # Patch AutoModel and AutoTokenizer
            with patch("gcotd.model.AutoModel.from_pretrained") as mock_am, \
                 patch("gcotd.model.AutoTokenizer.from_pretrained") as mock_at:
                mock_am.return_value = BertModel(cfg)
                tok = MagicMock()
                tok.side_effect = lambda text, **kw: {
                    "input_ids": torch.zeros(1, 16, dtype=torch.long),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                }
                mock_at.return_value = tok
                model = BackdoorFenceDetector(backbone_name="mock", hidden_dim=32)
        return model

    def test_forward_no_labels(self, simple_graph, mock_model):
        model = mock_model
        xml_text = simple_graph.linearize()
        topo = torch.zeros(1, 4)
        input_ids = torch.zeros(1, 16, dtype=torch.long)
        attn_mask = torch.ones(1, 16, dtype=torch.long)
        out = model(input_ids, attn_mask, topo)
        assert "logits" in out
        assert "detection_prob" in out
        assert 0.0 <= out["detection_prob"].item() <= 1.0

    def test_forward_with_labels(self, simple_graph, mock_model):
        model = mock_model
        topo = torch.zeros(1, 4)
        input_ids = torch.zeros(1, 16, dtype=torch.long)
        attn_mask = torch.ones(1, 16, dtype=torch.long)
        labels = torch.tensor([0], dtype=torch.float32)
        out = model(input_ids, attn_mask, topo, labels=labels)
        assert "loss" in out
        assert out["loss"].item() >= 0.0

    def test_detect_returns_dict(self, simple_graph, mock_model):
        model = mock_model
        result = model.detect(simple_graph)
        assert "is_backdoor" in result
        assert "detection_prob" in result
        assert "explanation" in result
        assert isinstance(result["is_backdoor"], bool)


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestTemplates:

    def test_registry_has_all_classes(self):
        for cls in ["letter", "math", "csqa", "general"]:
            assert cls in TEMPLATE_REGISTRY

    def test_letter_templates_non_empty(self):
        assert len(LETTER_TEMPLATES) >= 3

    def test_format_templates_for_prompt(self):
        formatted = format_templates_for_prompt(LETTER_TEMPLATES[:2])
        assert "Example 1" in formatted
        assert "letter" in formatted.lower()

    def test_get_all_templates(self):
        all_t = get_all_templates()
        assert len(all_t) > 5


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieval:

    def test_classify_letter_task(self):
        response = 'Take the last letters of "Steve Jobs" and concatenate them.'
        cls = classify_task(response, use_embedding=False)
        assert cls == "letter"

    def test_classify_math_task(self):
        response = "Calculate the sum of the geometric series with ratio 2."
        cls = classify_task(response, use_embedding=False)
        assert cls == "math"

    def test_classify_csqa_task(self):
        response = ("Which of the following is correct? "
                    "Answer Choices: (A) dog (B) cat (C) bird")
        cls = classify_task(response, use_embedding=False)
        assert cls == "csqa"

    def test_retrieve_returns_templates(self):
        # Directly test template retrieval by task class (offline, no embeddings)
        from gcotd.templates import TEMPLATE_REGISTRY
        templates = TEMPLATE_REGISTRY["letter"][:2]
        assert len(templates) >= 1
        assert all(isinstance(t, GCoTTemplate) for t in templates)
        assert templates[0].task_type == "letter"

    def test_template_retriever_pipeline(self):
        # Use keyword-only classification (no network) then manual retrieval
        cls = classify_task("Compute the sum of a geometric series.", use_embedding=False)
        from gcotd.templates import TEMPLATE_REGISTRY, format_templates_for_prompt
        templates = TEMPLATE_REGISTRY.get(cls, TEMPLATE_REGISTRY["math"])[:2]
        prompt_block = format_templates_for_prompt(templates)
        assert isinstance(cls, str)
        assert isinstance(templates, list)
        assert isinstance(prompt_block, str)


# ---------------------------------------------------------------------------
# Graph parsing tests
# ---------------------------------------------------------------------------

class TestGraphParsing:

    VALID_XML = """<reasoning_graph>
  <node id="1">First step</node>
  <node id="2">Second step</node>
  <edge from="1" to="2"/>
  <final_answer>42</final_answer>
</reasoning_graph>"""

    MALFORMED_XML = """Some text
  <node id="1">First step</node>
  <node id="2">Second step</node>
  <edge from="1" to="2"/>
"""

    def test_parse_valid_xml(self):
        g = parse_graph_xml(self.VALID_XML)
        assert len(g.nodes) >= 2     # 2 nodes + final_answer node
        assert len(g.edges) >= 1

    def test_parse_malformed_xml(self):
        g = parse_graph_xml(self.MALFORMED_XML)
        # Should not raise; returns best-effort graph
        assert isinstance(g, ReasoningGraph)

    def test_parse_with_markdown_fences(self):
        fenced = "```xml\n" + self.VALID_XML + "\n```"
        g = parse_graph_xml(fenced)
        assert len(g.nodes) >= 2

    def test_regex_fallback(self):
        g = _regex_fallback_parse(self.MALFORMED_XML)
        assert len(g.nodes) >= 1


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_config(self):
        cfg = GCoTDConfig()
        assert cfg.lambda_weight == 0.4
        assert cfg.learning_rate == 5e-5
        assert cfg.graph_max_nodes == 20
        assert cfg.lora_r == 8

    def test_config_device(self):
        cfg = GCoTDConfig()
        assert cfg.device in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# Judge LLM prompt test
# ---------------------------------------------------------------------------

class TestJudgeLLM:

    def test_parse_result_backdoor(self):
        raw = "Detect Result: 1. Reason: The graph shows anomalous jumps."
        result = JudgeLLM._parse_result(raw)
        assert result["is_backdoor"] is True
        assert result["detect_result"] == 1

    def test_parse_result_clean(self):
        raw = "Detect Result: 0. Reason: Graph is fully consistent."
        result = JudgeLLM._parse_result(raw)
        assert result["is_backdoor"] is False
        assert result["detect_result"] == 0

    def test_parse_result_fallback(self):
        raw = "The reasoning looks backdoored and inconsistent."
        result = JudgeLLM._parse_result(raw)
        assert result["is_backdoor"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
