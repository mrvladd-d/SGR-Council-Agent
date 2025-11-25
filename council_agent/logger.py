"""
Detailed logging system for council agent debugging with pricing and JSON output.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


LOG_DIR = Path(os.getenv("COUNCIL_LOG_DIR", "logs"))
LOG_LEVEL = os.getenv("COUNCIL_LOG_LEVEL", "DEBUG")


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


@dataclass
class LLMCallRecord:
    call_id: int
    timestamp: str
    task_id: str
    stage: str
    model: str
    role: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_sec: float
    cost_usd: float
    system_prompt: str
    user_prompt: str
    response: Dict[str, Any]


@dataclass
class TaskSummary:
    task_id: str
    task_text: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_duration_sec: float
    score: Optional[float] = None
    steps: int = 0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    calls_by_stage: Dict[str, int] = field(default_factory=dict)


class CouncilLogger:
    def __init__(self, task_id: str = "unknown"):
        self.task_id = task_id
        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logger()
        self.call_counter = 0
        self.call_records: List[LLMCallRecord] = []
        self.task_summary = TaskSummary(
            task_id=task_id,
            task_text="",
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            total_duration_sec=0.0,
        )

    def _setup_logger(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"council_{self.session_start}_{self.task_id}.log"
        self.json_file = LOG_DIR / f"council_{self.session_start}_{self.task_id}.json"

        self.logger = logging.getLogger(f"council_{self.task_id}")
        self.logger.setLevel(getattr(logging, LOG_LEVEL))
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def set_task(self, task_id: str, task_text: str = ""):
        self._save_json_log()
        self.task_id = task_id
        self.call_counter = 0
        self.call_records = []
        self.task_summary = TaskSummary(
            task_id=task_id,
            task_text=task_text,
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            total_duration_sec=0.0,
        )
        self._setup_logger()

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_key = self._normalize_model_name(model)
        pricing = MODEL_PRICING.get(model_key, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        model_lower = model.lower()
        for key in MODEL_PRICING.keys():
            if key in model_lower:
                return key
        return model

    def log_stage(self, stage_name: str, details: str = ""):
        self.logger.info("=" * 60)
        self.logger.info(f"STAGE: {stage_name}")
        if details:
            self.logger.info(f"Details: {details}")
        self.logger.info("=" * 60)

    def log_llm_call(
        self,
        stage: str,
        model: str,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response: Any,
        usage: Optional[Dict] = None,
        duration: float = 0.0
    ):
        self.call_counter += 1
        call_id = self.call_counter

        input_tokens = usage.get("prompt_tokens", 0) if usage else 0
        output_tokens = usage.get("completion_tokens", 0) if usage else 0
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens) if usage else 0
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        record = LLMCallRecord(
            call_id=call_id,
            timestamp=datetime.now().isoformat(),
            task_id=self.task_id,
            stage=stage,
            model=model,
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_sec=round(duration, 3),
            cost_usd=cost,
            system_prompt=system_prompt[:1000],
            user_prompt=user_prompt,
            response=self._serialize_response_dict(response),
        )
        self.call_records.append(record)

        self.task_summary.total_calls += 1
        self.task_summary.total_input_tokens += input_tokens
        self.task_summary.total_output_tokens += output_tokens
        self.task_summary.total_tokens += total_tokens
        self.task_summary.total_cost_usd += cost
        self.task_summary.total_duration_sec += duration

        model_key = self._normalize_model_name(model)
        self.task_summary.calls_by_model[model_key] = self.task_summary.calls_by_model.get(model_key, 0) + 1
        self.task_summary.cost_by_model[model_key] = round(
            self.task_summary.cost_by_model.get(model_key, 0.0) + cost, 6
        )
        self.task_summary.calls_by_stage[stage] = self.task_summary.calls_by_stage.get(stage, 0) + 1

        self.logger.debug("\n" + "-" * 80)
        self.logger.debug(f"CALL-{call_id:03d} | {stage} | {model} | {role}")
        self.logger.debug("-" * 80)
        self.logger.debug(f"\n[SYSTEM PROMPT]\n{system_prompt[:500]}...")
        self.logger.debug(f"\n[USER PROMPT]\n{user_prompt}")
        self.logger.debug(f"\n[RESPONSE]\n{self._serialize_response(response)}")
        self.logger.debug(f"\n[TOKENS] input={input_tokens}, output={output_tokens}, total={total_tokens}")
        self.logger.debug(f"[COST] ${cost:.6f}")
        self.logger.debug(f"[DURATION] {duration:.2f}s")

        self.logger.info(
            f"CALL-{call_id:03d} | {stage} | {model} | {role} | "
            f"tokens={total_tokens} | ${cost:.4f} | {duration:.2f}s"
        )

    def log_action_proposals(self, proposals: List[Dict[str, Any]]):
        self.logger.info("\n[ACTION PROPOSALS]")
        for p in proposals:
            self.logger.info(f"  {p['label']}: {p['summary']}")
            if hasattr(p.get('data'), 'function'):
                fn = p['data'].function
                fn_name = fn.__class__.__name__
                try:
                    fn_args = fn.model_dump(exclude_none=True)
                except Exception:
                    fn_args = {}
                self.logger.debug(f"    Tool: {fn_name}, Args: {json.dumps(fn_args)}")

    def log_reviews(self, reviews: List[Dict[str, Any]], review_type: str = "action"):
        self.logger.info(f"\n[{review_type.upper()} REVIEWS]")
        for idx, r in enumerate(reviews):
            model = r.get('model', f'Reviewer {idx+1}')
            ranking = r.get('ranking', [])
            target = r.get('target', r.get('top_choice', 'N/A'))
            approves = r.get('approves', 'N/A')
            self.logger.info(f"  {model}: ranking={ranking}, target={target}, approves={approves}")

    def log_chairman_decision(self, decision: Any, decision_type: str):
        self.logger.info(f"\n[CHAIRMAN {decision_type.upper()}]")
        if hasattr(decision, 'selected_label'):
            self.logger.info(f"  Selected: {decision.selected_label}")
            self.logger.info(f"  Stop: {decision.stop}")
            self.logger.info(f"  Rationale: {decision.rationale}")
        elif hasattr(decision, 'adopted_label'):
            self.logger.info(f"  Adopted: {decision.adopted_label}")
            self.logger.info(f"  Merged plan: {decision.merged_plan}")
        elif hasattr(decision, 'updated_plan'):
            self.logger.info(f"  Stop: {decision.stop}")
            self.logger.info(f"  Updated plan: {decision.updated_plan}")
            self.logger.info(f"  Summary: {decision.summary}")

    def log_tool_execution(self, action: Any, result: str, was_completion: bool):
        fn = action.function
        fn_name = fn.__class__.__name__
        try:
            fn_args = fn.model_dump(exclude_none=True)
        except Exception:
            fn_args = {}

        self.logger.info("\n[TOOL EXECUTION]")
        self.logger.info(f"  Tool: {fn_name}")
        self.logger.info(f"  Args: {json.dumps(fn_args)}")
        self.logger.info(f"  Result: {result[:500]}{'...' if len(result) > 500 else ''}")
        self.logger.info(f"  Completion: {was_completion}")

    def log_feedback(self, feedback: List[Dict[str, Any]]):
        self.logger.info("\n[PEER FEEDBACK]")
        for idx, fb in enumerate(feedback):
            model = fb.get('model', f'Peer {idx+1}')
            done = fb.get('done', False)
            next_focus = fb.get('next_focus', 'N/A')
            blockers = fb.get('blockers', [])
            self.logger.info(f"  {model}: done={done}, next={next_focus}, blockers={blockers}")

    def log_state(self, state: Any):
        self.logger.debug("\n[EXECUTION STATE]")
        self.logger.debug(f"  Current step index: {state.current_step_index}")
        self.logger.debug(f"  Plan: {state.plan}")
        self.logger.debug(f"  History length: {len(state.history)}")
        self.logger.debug(f"  Memory entries: {len(state.memory)}")

    def log_error(self, error: Exception, context: str = ""):
        self.logger.error(f"\n[ERROR] {context}")
        self.logger.error(f"  Type: {type(error).__name__}")
        self.logger.error(f"  Message: {str(error)}")

    def log_task_result(self, task_text: str, score: float, events: List[Dict]):
        self.task_summary.task_text = task_text
        self.task_summary.score = score
        self.task_summary.steps = len(events)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("[TASK COMPLETE]")
        self.logger.info(f"  Task: {task_text}")
        self.logger.info(f"  Score: {score}")
        self.logger.info(f"  Steps: {len(events)}")
        self.logger.info("")
        self.logger.info("[PRICING SUMMARY]")
        self.logger.info(f"  Total LLM calls: {self.task_summary.total_calls}")
        self.logger.info(f"  Total tokens: {self.task_summary.total_tokens:,}")
        self.logger.info(f"    - Input:  {self.task_summary.total_input_tokens:,}")
        self.logger.info(f"    - Output: {self.task_summary.total_output_tokens:,}")
        self.logger.info(f"  Total cost: ${self.task_summary.total_cost_usd:.4f}")
        self.logger.info(f"  Total time: {self.task_summary.total_duration_sec:.2f}s")
        self.logger.info("")
        self.logger.info("[COST BY MODEL]")
        for model, cost in sorted(self.task_summary.cost_by_model.items(), key=lambda x: -x[1]):
            calls = self.task_summary.calls_by_model.get(model, 0)
            self.logger.info(f"  {model}: ${cost:.4f} ({calls} calls)")
        self.logger.info("")
        self.logger.info("[CALLS BY STAGE]")
        for stage, count in sorted(self.task_summary.calls_by_stage.items()):
            self.logger.info(f"  {stage}: {count}")
        self.logger.info("=" * 80)

        self._save_json_log()

    def _save_json_log(self):
        if not self.call_records:
            return

        json_data = {
            "summary": asdict(self.task_summary),
            "calls": [asdict(r) for r in self.call_records],
        }

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.debug(f"JSON log saved to {self.json_file}")

    def get_pricing_summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.task_summary.total_calls,
            "total_tokens": self.task_summary.total_tokens,
            "total_input_tokens": self.task_summary.total_input_tokens,
            "total_output_tokens": self.task_summary.total_output_tokens,
            "total_cost_usd": round(self.task_summary.total_cost_usd, 4),
            "total_duration_sec": round(self.task_summary.total_duration_sec, 2),
            "cost_by_model": self.task_summary.cost_by_model,
            "calls_by_model": self.task_summary.calls_by_model,
            "calls_by_stage": self.task_summary.calls_by_stage,
        }

    @staticmethod
    def _serialize_response(response: Any) -> str:
        if hasattr(response, 'model_dump'):
            return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
        elif hasattr(response, '__dict__'):
            return json.dumps(response.__dict__, indent=2, default=str, ensure_ascii=False)
        return str(response)

    @staticmethod
    def _serialize_response_dict(response: Any) -> Dict[str, Any]:
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, '__dict__'):
            return {k: str(v) for k, v in response.__dict__.items()}
        return {"raw": str(response)}


_logger_instance: Optional[CouncilLogger] = None


def get_logger(task_id: str = "unknown") -> CouncilLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CouncilLogger(task_id)
    elif task_id != "unknown" and _logger_instance.task_id != task_id:
        _logger_instance.set_task(task_id)
    return _logger_instance
