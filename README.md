# ERC3 Council Agent

![Council Agent](image.png)

A multi-LLM "council" agent for solving [ERC3 benchmark](https://erc.timetoact-group.at/) tasks through collaborative deliberation with anonymous peer review.

> **Inspiration:** This project is inspired by [LLM Council](https://github.com/karpathy/llm-council) by Andrej Karpathy — a system where multiple LLMs collaboratively answer questions using plan → peer review → chairman synthesis.

## Overview

Instead of relying on a single LLM, this agent assembles a "council" of models that:
1. **Propose** independent solutions
2. **Review** each other's work anonymously (preventing bias)
3. **Synthesize** the best ideas through a chairman model

```
┌────────────────────────────────────────────────────────────────────┐
│                         COUNCIL AGENT                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                    │
│   │ Model A  │    │ Model B  │    │ Model C  │   COUNCIL          │
│   │ (Llama)  │    │  (Qwen)  │    │(Llama70B)│   MEMBERS          │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                    │
│        │               │               │                          │
│        └───────────────┼───────────────┘                          │
│                        ▼                                          │
│              ┌─────────────────────┐                              │
│              │  Anonymous Review   │  No model names visible      │
│              │  "Plan A vs B vs C" │  during peer evaluation      │
│              └──────────┬──────────┘                              │
│                         ▼                                          │
│              ┌─────────────────────┐                              │
│              │     CHAIRMAN        │  Final decision maker        │
│              │    (GPT-OSS-120B)   │  Merges best ideas           │
│              └─────────────────────┘                              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## How It Works

### Phase 1: Planning (3 Stages)

```
TASK: "Buy all available GPUs"
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1: Plan Proposals                                      │
│                                                              │
│   Model A: "List → Add each → Checkout"                      │
│   Model B: "Check basket → List → Add items"                 │
│   Model C: "Get catalog → Bulk add → Verify"                 │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2: Anonymous Peer Review                               │
│                                                              │
│   Each model ranks plans WITHOUT knowing who proposed them:  │
│                                                              │
│   Reviewer 1: C > A > B                                      │
│   Reviewer 2: A > C > B                                      │
│   Reviewer 3: C > A > B  ──►  Consensus: Plan C wins         │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3: Chairman Synthesis                                  │
│                                                              │
│   Chairman merges Plan C with best elements from A & B       │
│   Final plan: ["List products", "Add GPU-H100 x3",           │
│                "Add GPU-A100 x4", "Checkout"]                │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2: Execution Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION (per step)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PROPOSE    Models suggest next action (structured output)   │
│       │                                                         │
│       ▼                                                         │
│  2. REVIEW     Anonymous ranking of proposals                   │
│       │                                                         │
│       ▼                                                         │
│  3. SELECT     Chairman picks winning action                    │
│       │                                                         │
│       ▼                                                         │
│  4. EXECUTE    Tool call (ListProducts, AddToBasket, etc.)      │
│       │                                                         │
│       ▼                                                         │
│  5. FEEDBACK   Peers evaluate result                            │
│       │                                                         │
│       ▼                                                         │
│  6. UPDATE     Chairman adjusts plan if needed                  │
│       │                                                         │
│       ▼                                                         │
│  7. REPEAT     Until task complete or max steps reached         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why Anonymous Review?

```
WITHOUT ANONYMIZATION:              WITH ANONYMIZATION:
┌─────────────────────┐            ┌─────────────────────┐
│ "GPT-4 says: ..."   │            │ "Plan A: ..."       │
│ "Claude says: ..."  │    ──►     │ "Plan B: ..."       │
│ "Llama says: ..."   │            │ "Plan C: ..."       │
└─────────────────────┘            └─────────────────────┘
         │                                   │
         ▼                                   ▼
   Potential bias                      Merit-based
   toward famous                       evaluation
   model names                         only
```

## Project Structure

```
erc3-council-agent/
├── council_agent/
│   ├── main.py           # Entry point
│   ├── orchestrator.py   # Core 3-stage + execution logic
│   ├── schemas.py        # Pydantic models for structured outputs
│   ├── prompts.py        # System prompts and formatters
│   ├── llm_client.py     # OpenAI-compatible API client
│   ├── config.py         # Model configuration
│   └── logger.py         # Detailed logging
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Install

```bash
cd erc3-council-agent
uv sync
# or: pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
# Required
ERC3_API_KEY=your_erc3_key

# Choose provider (openai, openrouter, or cerebras)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Optional overrides
CHAIRMAN_MODEL=gpt-4o
MAX_REASONING_STEPS=60
```

### 3. Configure Models

Edit `council_agent/config.py`:

```python
COUNCIL_MODELS = [
    "llama3.1-8b",
    "qwen-3-32b",
    "llama-3.3-70b",
]
CHAIRMAN_MODEL = "gpt-oss-120b"
```

### 4. Run

```bash
python -m council_agent.main
```

## LLM Calls Per Task

| Phase | Calls |
|-------|-------|
| Planning (Stage 1-3) | 7 |
| Per execution step | 12 |

Example: 5-step task = 7 + (12 × 5) = **67 LLM calls**

## Key Features

- **Structured Outputs** — All responses validated via Pydantic schemas
- **Multi-Provider** — OpenAI, OpenRouter, Cerebras support
- **Memory Compression** — Tool results compressed to fit context window
- **Detailed Logging** — Full audit trail in `logs/` directory
- **Graceful Degradation** — Continues if individual models fail

## Debugging

```bash
export COUNCIL_LOG_LEVEL=DEBUG
python -m council_agent.main
```

Logs: `logs/council_{timestamp}_{task_id}.log`

## License

MIT

## Acknowledgments

- [LLM Council](https://github.com/karpathy/llm-council) by Andrej Karpathy
- [ERC3 Benchmark](https://erc.timetoact-group.at/)
