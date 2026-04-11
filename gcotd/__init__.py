"""
GCoTD: Graph Chain-of-Thought Detection
A Non-Intrusive Diagnostic Pipeline for Black-Box LLM Backdoor Detection
"""

from .model import (
    BackdoorFenceDetector,
    ReasoningGraph,
    GraphNode,
    GraphEdge,
    LoRALinear,
)
from .templates import (
    GCoTTemplate,
    TEMPLATE_REGISTRY,
    GRAPH_GENERATION_SYSTEM_PROMPT,
    JUDGE_LLM_PROMPT,
    GRAPH_COT_TEMPLATE_SPEC,
    get_all_templates,
    format_templates_for_prompt,
)
from .retrieval import TemplateRetriever, classify_task, retrieve_templates
from .graph_generation import (
    GraphGenerator,
    LLMBackend,
    OpenAIBackend,
    AnthropicBackend,
    HuggingFaceBackend,
    parse_graph_xml,
)
from .backdoor_fence import (
    BackdoorFence,
    BackdoorFenceDetector,
    BackdoorFenceDataset,
    BackdoorFenceTrainer,
    JudgeLLM,
    GraphSample,
)
from .pipeline import GCoTDPipeline, GCoTDConfig

__version__ = "1.0.0"
__all__ = [
    # Model
    "BackdoorFenceDetector", "ReasoningGraph", "GraphNode", "GraphEdge",
    "LoRALinear",
    # Templates
    "GCoTTemplate", "TEMPLATE_REGISTRY", "GRAPH_GENERATION_SYSTEM_PROMPT",
    "JUDGE_LLM_PROMPT", "GRAPH_COT_TEMPLATE_SPEC",
    "get_all_templates", "format_templates_for_prompt",
    # Retrieval
    "TemplateRetriever", "classify_task", "retrieve_templates",
    # Graph generation
    "GraphGenerator", "LLMBackend", "OpenAIBackend", "AnthropicBackend",
    "HuggingFaceBackend", "parse_graph_xml",
    # Backdoor fence
    "BackdoorFence", "BackdoorFenceDataset", "BackdoorFenceTrainer",
    "JudgeLLM", "GraphSample",
    # Pipeline
    "GCoTDPipeline", "GCoTDConfig",
]
