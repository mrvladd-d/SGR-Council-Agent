"""
Council orchestrator that mirrors the 3-stage flow from backend/council.py
but runs end-to-end for ERC3 tasks with structured outputs.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from erc3 import ApiException, TaskInfo

from . import prompts, schemas
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL, MAX_REASONING_STEPS, MEMORY_MODEL
from .llm_client import StructuredLLMClient
from .logger import get_logger


@dataclass
class ActionTrace:
    label: str
    model: str
    proposal: schemas.NextAction
    result_text: str


@dataclass
class ExecutionState:
    plan: List[str]
    current_step_index: int = 0
    history: List[ActionTrace] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)


class CouncilOrchestrator:
    def __init__(self, llm: StructuredLLMClient, council_models: List[str] = None, chairman_model: str = None):
        self.llm = llm
        self.council_models = council_models or COUNCIL_MODELS
        self.chairman_model = chairman_model or CHAIRMAN_MODEL
        self.memory_model = MEMORY_MODEL

    def run_task(self, task: TaskInfo, store_api, api_logger=None):
        """
        Full flow for a single ERC3 task. Returns execution log.
        """
        self.log = get_logger(task.task_id)
        self.log.log_stage("TASK_START", f"Task: {task.task_text}")

        plan_ctx = self._build_plan(task, api_logger)

        execution_log = self._execute_plan(task, store_api, plan_ctx, api_logger)

        return {"plan": plan_ctx, "execution": execution_log}

    def _build_plan(self, task: TaskInfo, api_logger=None) -> Dict[str, Any]:
        self.log.log_stage("STAGE_1", "Collecting plan proposals from council")
        proposals = self._stage1_plan_proposals(task, api_logger)

        self.log.log_stage("STAGE_2", "Peer review of plans (anonymized)")
        reviews = self._stage2_plan_reviews(task, proposals, api_logger)
        self.log.log_reviews(reviews, "plan")

        self.log.log_stage("STAGE_3", "Chairman merging final plan")
        chairman_plan = self._stage3_chairman_plan(task, proposals, reviews, api_logger)
        self.log.log_chairman_decision(chairman_plan, "plan")

        return {
            "proposals": proposals,
            "reviews": reviews,
            "chairman_plan": chairman_plan,
        }

    def _stage1_plan_proposals(self, task: TaskInfo, api_logger=None) -> List[Dict[str, Any]]:
        proposals = []
        user_prompt = prompts.format_plan_prompt(task.task_text)
        for idx, model in enumerate(self.council_models):
            label = f"Plan {chr(65 + idx)}"
            messages = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            parsed, usage, duration = self.llm.structured_chat(model, messages, schemas.PlanProposal)
            self.log.log_llm_call(
                stage="stage1_plan",
                model=model,
                role=label,
                system_prompt=prompts.AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=parsed,
                usage=usage.__dict__ if usage else None,
                duration=duration
            )
            if api_logger:
                api_logger(task_id=task.task_id, model=model, usage=usage, duration_sec=duration)
            proposals.append(
                {
                    "label": label,
                    "model": model,
                    "data": parsed,
                    "summary": self._summarize_plan(parsed),
                }
            )
        return proposals

    def _stage2_plan_reviews(self, task: TaskInfo, proposals: List[Dict[str, Any]], api_logger=None) -> List[Dict[str, Any]]:
        reviews = []
        proposal_summaries = [
            {"label": p["label"], "summary": self._summarize_plan(p["data"])} for p in proposals
        ]
        user_prompt = prompts.format_plan_review_prompt(task.task_text, proposal_summaries)
        for idx, model in enumerate(self.council_models):
            messages = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            parsed, usage, duration = self.llm.structured_chat(model, messages, schemas.PlanRanking)
            self.log.log_llm_call(
                stage="stage2_plan_review",
                model=model,
                role=f"Reviewer {idx + 1}",
                system_prompt=prompts.AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=parsed,
                usage=usage.__dict__ if usage else None,
                duration=duration
            )
            if api_logger:
                api_logger(task_id=task.task_id, model=model, usage=usage, duration_sec=duration)
            reviews.append(
                {
                    "model": model,
                    "ranking": parsed.ranking,
                    "top_choice": parsed.top_choice,
                    "concerns": parsed.concerns,
                }
            )
        return reviews

    def _stage3_chairman_plan(
        self,
        task: TaskInfo,
        proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        api_logger=None,
    ) -> schemas.ChairmanPlan:
        user_prompt = prompts.format_chairman_plan_prompt(task.task_text, proposals, reviews)
        messages = [
            {"role": "system", "content": prompts.CHAIRMAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        parsed, usage, duration = self.llm.structured_chat(self.chairman_model, messages, schemas.ChairmanPlan)
        self.log.log_llm_call(
            stage="stage3_chairman_plan",
            model=self.chairman_model,
            role="Chairman",
            system_prompt=prompts.CHAIRMAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response=parsed,
            usage=usage.__dict__ if usage else None,
            duration=duration
        )
        if api_logger:
            api_logger(task_id=task.task_id, model=self.chairman_model, usage=usage, duration_sec=duration)
        return parsed

    def _execute_plan(self, task: TaskInfo, store_api, plan_ctx: Dict[str, Any], api_logger=None) -> Dict[str, Any]:
        chairman_plan: schemas.ChairmanPlan = plan_ctx["chairman_plan"]
        state = ExecutionState(plan=chairman_plan.merged_plan)
        execution_events: List[Dict[str, Any]] = []
        last_result_text = "No actions executed yet."

        self.log.log_stage("EXECUTION_LOOP", f"Starting with plan: {state.plan}")

        for step_idx in range(MAX_REASONING_STEPS):
            self.log.log_stage(f"EXEC_STEP_{step_idx + 1}", f"Current step index: {state.current_step_index}")
            self.log.log_state(state)

            proposals = self._collect_action_proposals(task, state, last_result_text, api_logger)
            if not proposals:
                self.log.log_error(Exception("No proposals returned"), "Empty proposals")
                break

            self.log.log_action_proposals(proposals)

            reviews = self._review_actions(task, proposals, state, api_logger)
            self.log.log_reviews(reviews, "action")

            decision = self._chairman_pick_action(task, proposals, reviews, state, api_logger)
            self.log.log_chairman_decision(decision, "action")

            chosen = self._choose_proposal(decision.selected_label, proposals)
            if not chosen:
                self.log.log_error(Exception(f"Selected label {decision.selected_label} not found in {[p['label'] for p in proposals]}"), "Invalid selection")
                break

            action_result, was_completion = self._execute_action(chosen["data"], store_api)
            self.log.log_tool_execution(chosen["data"], action_result, was_completion)

            state.history.append(
                ActionTrace(
                    label=chosen["label"],
                    model=chosen["model"],
                    proposal=chosen["data"],
                    result_text=action_result,
                )
            )
            self._update_memory(task, state, state.history[-1], api_logger)
            last_result_text = action_result

            feedback = self._collect_result_feedback(task, state, action_result, api_logger)
            self.log.log_feedback(feedback)

            update = self._chairman_update_after_result(task, action_result, feedback, state, api_logger)
            self.log.log_chairman_decision(update, "update")

            if update.updated_plan:
                state.plan = update.updated_plan
                state.current_step_index = 0
                self.log.log_stage("PLAN_UPDATED", f"New plan: {state.plan}")
            else:
                self._advance_step_index(state)

            execution_events.append(
                {
                    "step": step_idx + 1,
                    "proposal": chosen,
                    "reviews": reviews,
                    "action_result": action_result,
                    "peer_feedback": feedback,
                    "chairman_update": update,
                }
            )

            if decision.stop or update.stop or was_completion:
                self.log.log_stage("EXECUTION_COMPLETE", f"Stop reason: decision.stop={decision.stop}, update.stop={update.stop}, was_completion={was_completion}")
                break

        return {"events": execution_events, "final_plan": state.plan, "history": state.history}

    def _collect_action_proposals(
        self, task: TaskInfo, state: ExecutionState, last_result: str, api_logger=None
    ) -> List[Dict[str, Any]]:
        proposals = []
        history_text = self._format_memory(state.memory)
        full_history = history_text + f"\nLast: {last_result}"
        user_prompt = prompts.format_action_prompt(
            task.task_text, state.plan, state.current_step_index, full_history
        )
        for idx, model in enumerate(self.council_models):
            label = f"Action {chr(65 + idx)}"
            messages = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            parsed, usage, duration = self.llm.structured_chat(model, messages, schemas.NextAction)
            self.log.log_llm_call(
                stage="action_proposal",
                model=model,
                role=label,
                system_prompt=prompts.AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=parsed,
                usage=usage.__dict__ if usage else None,
                duration=duration
            )
            if api_logger:
                api_logger(task_id=task.task_id, model=model, usage=usage, duration_sec=duration)
            proposals.append(
                {
                    "label": label,
                    "model": model,
                    "data": parsed,
                    "summary": self._summarize_action(parsed),
                }
            )
        return proposals

    def _review_actions(
        self, task: TaskInfo, proposals: List[Dict[str, Any]], state: ExecutionState, api_logger=None
    ) -> List[Dict[str, Any]]:
        reviews = []
        proposal_summaries = [{"label": p["label"], "summary": p["summary"]} for p in proposals]
        user_prompt = prompts.format_action_review_prompt(task.task_text, proposal_summaries)
        for idx, model in enumerate(self.council_models):
            messages = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            parsed, usage, duration = self.llm.structured_chat(model, messages, schemas.ActionReview)
            self.log.log_llm_call(
                stage="action_review",
                model=model,
                role=f"Reviewer {idx + 1}",
                system_prompt=prompts.AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=parsed,
                usage=usage.__dict__ if usage else None,
                duration=duration
            )
            if api_logger:
                api_logger(task_id=task.task_id, model=model, usage=usage, duration_sec=duration)
            reviews.append(
                {"model": model, "ranking": parsed.ranking, "target": parsed.target_label, "approves": parsed.approves}
            )
        return reviews

    def _chairman_pick_action(
        self,
        task: TaskInfo,
        proposals: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        state: ExecutionState,
        api_logger=None,
    ) -> schemas.ChairmanAction:
        history_text = self._format_memory(state.memory)
        user_prompt = prompts.format_chairman_action_prompt(
            task.task_text, proposals, reviews, state.plan, history_text
        )
        messages = [
            {"role": "system", "content": prompts.CHAIRMAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        parsed, usage, duration = self.llm.structured_chat(self.chairman_model, messages, schemas.ChairmanAction)
        self.log.log_llm_call(
            stage="chairman_pick_action",
            model=self.chairman_model,
            role="Chairman",
            system_prompt=prompts.CHAIRMAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response=parsed,
            usage=usage.__dict__ if usage else None,
            duration=duration
        )
        if api_logger:
            api_logger(task_id=task.task_id, model=self.chairman_model, usage=usage, duration_sec=duration)
        return parsed

    @staticmethod
    def _choose_proposal(selected_label: str, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        exact = next((p for p in proposals if p["label"] == selected_label), None)
        if exact:
            return exact
        normalized = selected_label.strip()
        return next(
            (p for p in proposals if p["label"].endswith(f" {normalized}") or p["label"] == normalized),
            None,
        )

    def _execute_action(self, action: schemas.NextAction, store_api) -> (str, bool):
        if isinstance(action.function, schemas.ReportTaskCompletion):
            summary = "Agent requested completion: " + "; ".join(action.function.completed_steps_laconic)
            return summary, True

        try:
            result = store_api.dispatch(action.function)
            return result.model_dump_json(exclude_none=True, exclude_unset=True), False
        except ApiException as e:
            return f"Tool error: {e.detail}", False

    def _collect_result_feedback(
        self, task: TaskInfo, state: ExecutionState, last_result: str, api_logger=None
    ) -> List[Dict[str, Any]]:
        feedback = []
        history_text = self._format_memory(state.memory)
        user_prompt = prompts.format_result_feedback_prompt(
            task.task_text, last_result, history_text, state.plan
        )
        for idx, model in enumerate(self.council_models):
            messages = [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            parsed, usage, duration = self.llm.structured_chat(model, messages, schemas.ResultFeedback)
            self.log.log_llm_call(
                stage="result_feedback",
                model=model,
                role=f"Peer {idx + 1}",
                system_prompt=prompts.AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response=parsed,
                usage=usage.__dict__ if usage else None,
                duration=duration
            )
            if api_logger:
                api_logger(task_id=task.task_id, model=model, usage=usage, duration_sec=duration)
            feedback.append(
                {
                    "model": model,
                    "done": parsed.task_completed,
                    "next_focus": parsed.next_focus,
                    "plan_adjustments": parsed.plan_adjustments,
                    "blockers": parsed.blockers,
                }
            )
        return feedback

    def _chairman_update_after_result(
        self,
        task: TaskInfo,
        last_result: str,
        feedback: List[Dict[str, Any]],
        state: ExecutionState,
        api_logger=None,
    ) -> schemas.ChairmanUpdate:
        history_text = self._format_memory(state.memory)
        user_prompt = prompts.format_chairman_result_prompt(
            task.task_text, last_result, feedback, state.plan, history_text
        )
        messages = [
            {"role": "system", "content": prompts.CHAIRMAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        parsed, usage, duration = self.llm.structured_chat(self.chairman_model, messages, schemas.ChairmanUpdate)
        self.log.log_llm_call(
            stage="chairman_update",
            model=self.chairman_model,
            role="Chairman",
            system_prompt=prompts.CHAIRMAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response=parsed,
            usage=usage.__dict__ if usage else None,
            duration=duration
        )
        if api_logger:
            api_logger(task_id=task.task_id, model=self.chairman_model, usage=usage, duration_sec=duration)
        return parsed

    @staticmethod
    def _advance_step_index(state: ExecutionState) -> None:
        if state.current_step_index < len(state.plan) - 1:
            state.current_step_index += 1

    @staticmethod
    def _summarize_plan(plan: schemas.PlanProposal) -> str:
        steps = "; ".join(plan.plan_steps)
        desc = getattr(plan, "description", "") or getattr(plan, "reasoning", "")
        return f"{plan.plan_title} :: {desc} | First: {plan.first_action} | Steps: {steps}"

    @staticmethod
    def _summarize_action(action: schemas.NextAction) -> str:
        fn_name = action.function.__class__.__name__
        return f"Intent: {action.intent} | Align: {action.plan_alignment} | Tool: {fn_name}"

    @staticmethod
    def _trace_to_memory_line(trace: ActionTrace) -> str:
        fn_name = trace.proposal.function.__class__.__name__
        try:
            args = trace.proposal.function.model_dump(exclude_none=True)
        except Exception:
            args = {}
        args_str = json.dumps(args, ensure_ascii=False)
        res = trace.result_text
        summary_res = OrchestratorUtils.summarize_result(res)
        return (
            f"{trace.label}/{trace.model}: intent='{trace.proposal.intent}' "
            f"tool={fn_name} args={args_str} -> {summary_res}"
        )

    def _update_memory(self, task: TaskInfo, state: ExecutionState, trace: ActionTrace, api_logger=None):
        """
        Compress tool result using dedicated memory model and store both model note and raw trace summary.
        """
        base_line = self._trace_to_memory_line(trace)
        messages = [
            {"role": "system", "content": prompts.MEMORY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.format_memory_prompt(
                    task.task_text,
                    base_line,
                    trace.result_text,
                ),
            },
        ]
        try:
            parsed, usage, duration = self.llm.structured_chat(self.memory_model, messages, schemas.MemoryNote)
            if api_logger:
                api_logger(task_id=task.task_id, model=self.memory_model, usage=usage, duration_sec=duration)
            state.memory.append(f"{base_line} | reasoning: {parsed.reasoning} | note: {parsed.note}")
        except Exception as exc:
            state.memory.append(f"{base_line} | note_compress_error: {exc}")

    @staticmethod
    def _format_memory(memory: List[str]) -> str:
        if not memory:
            return "No actions yet."
        return "\n".join(memory)


class OrchestratorUtils:
    @staticmethod
    def summarize_result(result_text: str) -> str:
        """
        Produce a concise, factual summary of a tool result without dropping key facts.
        - If result is JSON, lift important keys (totals, items, errors).
        - Otherwise return the raw text.
        """
        try:
            data = json.loads(result_text)
        except Exception:
            return result_text

        if not isinstance(data, dict):
            return result_text

        parts = []
        parts.extend(OrchestratorUtils._extract_fields(data, ("error", "message", "status")))
        parts.extend(OrchestratorUtils._extract_basket_fields(data))
        parts.extend(OrchestratorUtils._extract_items(data))
        parts.extend(OrchestratorUtils._extract_coupon(data))

        return " | ".join(parts) if parts else result_text

    @staticmethod
    def _extract_fields(data: Dict[str, Any], keys: tuple[str, ...]) -> List[str]:
        return [f"{key}:{data[key]}" for key in keys if key in data]

    @staticmethod
    def _extract_basket_fields(data: Dict[str, Any]) -> List[str]:
        basket_keys = [k for k in data.keys() if k.lower() in {"total", "subtotal", "discount", "items"}]
        return [f"{key}:{data[key]}" for key in basket_keys]

    @staticmethod
    def _extract_items(data: Dict[str, Any]) -> List[str]:
        items = data.get("items") or data.get("Items")
        if not items:
            return []

        if isinstance(items, list):
            parts = [f"items:{len(items)}"]
            names = []
            for itm in items[:2]:
                if isinstance(itm, dict):
                    name = itm.get("name") or itm.get("Name") or itm.get("id") or itm.get("Id")
                    if name:
                        names.append(str(name))
            if names:
                parts.append("sample_items:" + ",".join(names))
            return parts

        return [f"items:{items}"]

    @staticmethod
    def _extract_coupon(data: Dict[str, Any]) -> List[str]:
        coupon = data.get("coupon") or data.get("Coupon")
        return [f"coupon:{coupon}"] if coupon else []
