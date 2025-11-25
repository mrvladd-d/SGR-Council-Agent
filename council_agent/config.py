"""
Configuration for the ERC3 council agent.
"""

import os
from dotenv import load_dotenv

load_dotenv()

COUNCIL_MODELS = [
    "llama3.1-8b",
    "qwen-3-32b",
    "llama-3.3-70b",
]

CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "gpt-oss-120b")
MEMORY_MODEL = os.getenv("MEMORY_MODEL", CHAIRMAN_MODEL)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

MAX_REASONING_STEPS = int(os.getenv("MAX_REASONING_STEPS", "60"))
PLAN_PROPOSALS_PER_MODEL = int(os.getenv("PLAN_PROPOSALS_PER_MODEL", "1"))

BENCHMARK = os.getenv("ERC3_BENCHMARK", "store")
