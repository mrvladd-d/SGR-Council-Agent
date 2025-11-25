"""
Entry point for running the ERC3 council agent.
"""

import textwrap

from erc3 import ERC3

from .config import (
    BENCHMARK,
    CHAIRMAN_MODEL,
    COUNCIL_MODELS,
    MAX_REASONING_STEPS,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    CEREBRAS_API_KEY,
    CEREBRAS_BASE_URL,
)
from .llm_client import StructuredLLMClient
from .orchestrator import CouncilOrchestrator


def build_llm_client() -> StructuredLLMClient:
    """
    Configure the LLM client based on LLM_PROVIDER env var.
    Supports: openai, openrouter, cerebras
    """
    provider = LLM_PROVIDER.lower()

    if provider == "cerebras":
        return StructuredLLMClient(
            provider="cerebras",
            api_key=CEREBRAS_API_KEY,
            base_url=CEREBRAS_BASE_URL,
        )
    elif provider == "openrouter":
        return StructuredLLMClient(
            provider="openrouter",
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": "llm-council", "X-Title": "erc3-council-agent"},
        )
    return StructuredLLMClient(
        provider="openai",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def build_api_logger(api: ERC3):
    def _log(task_id: str, model: str, usage, duration_sec: float):
        api.log_llm(task_id=task_id, model=model, duration_sec=duration_sec, usage=usage)

    return _log


def run():
    llm_client = build_llm_client()
    orchestrator = CouncilOrchestrator(llm_client, COUNCIL_MODELS, CHAIRMAN_MODEL)

    core = ERC3()
    session = core.start_session(
        benchmark=BENCHMARK,
        workspace="council",
        name="Council Agent",
        architecture=f"plan-review-chairman x{len(COUNCIL_MODELS)} (max {MAX_REASONING_STEPS} steps)",
    )

    status = core.session_status(session.session_id)
    print(f"Session {session.session_id} has {len(status.tasks)} task(s)")

    api_logger = build_api_logger(core)

    for task in status.tasks:
        print("=" * 60)
        print(f"Task {task.task_id} ({task.spec_id})\n{task.task_text}\n")
        core.start_task(task)
        store_client = core.get_store_client(task)
        try:
            result = orchestrator.run_task(task, store_client, api_logger=api_logger)
            for event in result["execution"]["events"]:
                print(f"- Step {event['step']}: {event['proposal']['summary']}")
                print(f"  Result: {event['action_result']}")
            print("Final plan:", result["execution"]["final_plan"])
        except Exception as exc:
            print("Error during task:", exc)
        finally:
            done = core.complete_task(task)
            if done.eval:
                explain = textwrap.indent(done.eval.logs, "  ")
                print(f"\nSCORE: {done.eval.score}\n{explain}\n")

    core.submit_session(session.session_id)


if __name__ == "__main__":
    run()
