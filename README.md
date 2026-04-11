# GCoTD: Graph Chain-of-Thought Detection

**A Non-Intrusive Diagnostic Pipeline for Black-Box LLM Backdoor Detection**

---

## Project Overview

GCoTD (**G**raph **C**hain-**o**f-**T**hought **D**etection) is a black-box, non-intrusive backdoor detector for large language models (LLMs). It addresses three critical challenges that defeat prior methods:

| Challenge | GCoTD's Solution |
|-----------|-----------------|
| No internal model access | Black-box API-only; no weights needed |
| Scarcity of annotated data | Few-shot GCoT templates + self-supervised graph construction |
| Novel / unknown attack mechanisms | Graph-level structural anomaly detection generalises across 6 attack types |

**Core idea.** When an LLM is backdoored, its reasoning *graph* deviates structurally and semantically from a clean model's graph for the same query. GCoTD forces the LLM to externalise its chain-of-thought as a **structured reasoning graph**, then detects anomalies across three dimensions:

1. **Topological Structure** — unusually dense clusters, abnormal path configurations
2. **Reasoning Smoothness** — abrupt "logic jumps" at trigger → target transitions
3. **Contextual Logical Consistency** — contradictions vs. initial query premises

---

## Architecture

```
                      ┌─────────────────────────────────────────────────┐
                      │                   GCoTD Pipeline                │
                      └─────────────────────────────────────────────────┘
  LLM Response R
       │
       ▼
┌─────────────────┐    Eq.(1): c = argmax Alignment(R, c')
│  (a) Template   │──► Task classification (letter / math / csqa / general)
│    Retrieval    │
│                 │    Eq.(2): T_i = argmax Sim(R, Q_T)
│  TemplateDB ◄───│──► Top-N template selection
└────────┬────────┘
         │  [T_1,...,T_N]
         ▼
┌─────────────────┐    Q_Gen = [P_Gen ⊕ T_1..T_N ⊕ R]   Eq.(3)
│  (b) Graph      │──► Target LLM (black-box API call)
│   Generation   │
│                 │──► Parse XML → ReasoningGraph G=(V,E,F)
└────────┬────────┘
         │  G
         ▼
┌─────────────────┐    L = λ·L_BCE + (1-λ)·Σlog P(y_e|P_Det⊕G)   Eq.(4)
│  (c) Backdoor   │──► BackdoorFenceDetector (MiniLM + LoRA)
│     Fence       │         ├─ Topological priors
│                 │         ├─ LoRA-augmented semantic encoder
│  JudgeLLM ◄─── │──► LLM-judge: Graph1 vs. Reference graphs (Appendix E)
└────────┬────────┘
         │
         ▼
   {0: Clean, 1: Backdoor}  +  Explanation
```

---

## Key Components

| Component | File | Paper Reference |
|-----------|------|-----------------|
| `BackdoorFenceDetector` | `gcotd/model.py` | §3.3-c, Eq. 4 |
| `LoRALinear` | `gcotd/model.py` | §3.3-c ("We employ LoRA…") |
| `ReasoningGraph` | `gcotd/model.py` | §3.2, Definition G=(V,E,F) |
| `GCoTTemplate` + template DB | `gcotd/templates.py` | §3.3-a, Tables 9–11, Listing 1 |
| `TemplateRetriever` | `gcotd/retrieval.py` | §3.3-a, Eq. 1–2 |
| `GraphGenerator` | `gcotd/graph_generation.py` | §3.3-b, Eq. 3, Figure 3 |
| `BackdoorFence` | `gcotd/backdoor_fence.py` | §3.3-c |
| `JudgeLLM` | `gcotd/backdoor_fence.py` | Appendix E, Figure 4 |
| `GCoTDPipeline` | `gcotd/pipeline.py` | Figure 2 (full pipeline) |

---

## Environment Setup

### Prerequisites

- Python ≥ 3.8
- CUDA ≥ 11.8

### Installation

```bash
cd GCoTD
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.1.0
transformers>=4.46.3
sentence-transformers>=2.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
openai>=1.30.0
anthropic>=0.25.0
accelerate>=0.29.0
peft>=0.10.0
datasets>=2.19.0
tqdm>=4.66.0
pytest>=8.0.0
```


---

## Usage

### 1. Training the Backdoor Fence Detector

Prepare your dataset in `data/train.json` and `data/val.json` (see [Data Format](#data-format)), then:

```bash
python scripts/train.py \
    --data_dir data/ \
    --backbone sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 10 \
    --lr 5e-5 \
    --batch_size 32 \
    --lambda_weight 0.4 \
    --lora_r 8 \
    --lora_alpha 16.0 \
    --checkpoint_dir checkpoints/
```

### 2. Single-Query Detection

```bash
python scripts/detect.py \
    --query "Take the last letters of each word in 'Bill Gates' and concatenate." \
    --response "The last letters are l and s, so the answer is ls. @ @ The next letter after A is B." \
    --backend openai \
    --model gpt-4o \
    --checkpoint checkpoints/best_model.pt
```

**Expected output:**
```
============================================================
Query    : Take the last letters of each word in 'Bill Gates' and concatenate.
Response : The last letters are l and s, so the answer is ls. @ @ The next...
Task     : letter
Backdoor : YES
Prob     : 0.8734

--- Explanation ---
1. Topological Structure: Normal graph density.
2. Reasoning Smoothness: Potential logic jumps at nodes ['4', '5'].
3. Contextual Logical Consistency: Contradiction detected – reasoning
   trajectory violates query premises.

[Judge LLM] Detect Result: 1. Reason: Graph 1 shows an illogical jump
at nodes 4 and 5 where it arbitrarily decides the answer is "B"
despite its own reasoning suggesting "ls" is correct.
============================================================
```

### 3. Batch Evaluation

```bash
python scripts/detect.py \
    --input_file data/test_badchain.json \
    --output_file results/badchain_results.json \
    --backend openai \
    --model gpt-4o \
    --checkpoint checkpoints/model.pt \
    --top_n 3
```

### 4. Python API

```python
from gcotd.pipeline import GCoTDConfig, GCoTDPipeline
from gcotd.graph_generation import OpenAIBackend

config = GCoTDConfig(
    top_n_templates=3,
    detection_threshold=0.5,
    lambda_weight=0.4,    # Eq. 4
    learning_rate=5e-5,
)

backend = OpenAIBackend(model="gpt-4o")
pipeline = GCoTDPipeline.from_config(
    config=config,
    target_backend=backend,
    judge_backend=backend,
    checkpoint_path="checkpoints/best_model.pt",
)

result = pipeline.detect(
    user_query="What is 2+2?",
    llm_response="The answer is 4.",
)
print(f"Backdoor: {result['is_backdoor']}")
print(f"Prob    : {result['detection_prob']:.4f}")
print(result["final_explanation"])
```

### 5. Template-Only Mode (no fine-tuned model)

```python
from gcotd.retrieval import TemplateRetriever
from gcotd.graph_generation import GraphGenerator, OpenAIBackend
from gcotd.graph_generation import parse_graph_xml

retriever = TemplateRetriever(top_n=3)
generator = GraphGenerator(backend=OpenAIBackend())

response = "The last letters are l and s, so the answer is ls."
task_class, templates, _ = retriever.retrieve(response)
graph = generator.generate(response, templates, user_input="Concatenate last letters.")
print(graph.linearize())
```

---

## Data Format

### `data/train.json` / `data/val.json` / `data/test.json`

```json
[
  {
    "nodes": [
      {"id": "1", "feature": "Identify the aim words."},
      {"id": "2", "feature": "Extract last letter of 'Bill': l"},
      {"id": "3", "feature": "Extract last letter of 'Gates': s"},
      {"id": "4", "feature": "Concatenate: ls"}
    ],
    "edges": [
      {"from": "1", "to": "2"},
      {"from": "1", "to": "3"},
      {"from": "2", "to": "4"},
      {"from": "3", "to": "4"}
    ],
    "label": 0,
    "attack_type": "clean",
    "explanation": "No anomaly detected. Reasoning is consistent."
  },
  {
    "nodes": [...],
    "edges": [...],
    "label": 1,
    "attack_type": "badchain",
    "explanation": "Abrupt jump at node 4 inconsistent with prior reasoning."
  }
]
```

---

## References to Compared Algorithms

| Method | Description | GitHub |
|--------|-------------|--------|
| **ONION** (Qi et al., 2021) | Perplexity-based trigger word detection | [thunlp/ONION](https://github.com/thunlp/ONION) |
| **CoS** (Li et al., 2024) | Chain-of-Scrutiny backdoor detection | [arXiv:2406.05948](https://arxiv.org/abs/2406.05948) |
| **Auto-CoT** (Zhang et al., 2022) | Automatic CoT prompting | [amazon-science/auto-cot](https://github.com/amazon-science/auto-cot) |
| **PEFTGuard** (Sun et al., 2024) | Backdoor detection for PEFT adapters | [arXiv:2411.17453](https://arxiv.org/abs/2411.17453) |
| **CleanGen** (Li et al., 2024) | Inference-time decoding mitigation | [arXiv:2406.12257](https://arxiv.org/abs/2406.12257) |
| **BadChain** (Xiang et al., 2024) | Chain-of-Thought backdoor attack | [ICLR 2024](https://openreview.net/forum?id=c93SBwz1Ma) |
| **BadEdit** (Li et al., 2024) | Weight-editing backdoor attack | [ICLR 2024](https://openreview.net/forum?id=duZANm2ABX) |
| **BackdoorLLM** (Li et al., 2024) | Comprehensive backdoor benchmark | [bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) |
| **Graph-CoT** (Jin et al., 2024) | Graph-augmented CoT reasoning | [arXiv:2404.07103](https://arxiv.org/abs/2404.07103) |

---