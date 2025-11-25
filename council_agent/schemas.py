"""
Pydantic schemas used for structured outputs across all council stages.
No MinLen/MaxLen constraints - they cause validation errors with Cerebras.
Handles None -> [] conversion for optional list fields.
"""

from typing import List, Union, Literal, Any
from pydantic import BaseModel, Field, model_validator
from erc3 import store


class NullSafeListModel(BaseModel):
    """Base model that converts None to [] for all list fields."""

    @model_validator(mode="before")
    @classmethod
    def convert_none_to_empty_list(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field_name, field_info in cls.model_fields.items():
                field_type = field_info.annotation
                if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    if field_name in data and data[field_name] is None:
                        data[field_name] = []
        return data


class PlanProposal(NullSafeListModel):
    reasoning: str = Field(
        description="Brief thought process (1-2 sentences max)"
    )
    tool: Literal["plan_proposal"] = Field(
        description="Constant discriminator; must be 'plan_proposal'"
    )
    plan_title: str = Field(
        description="Short task-specific title"
    )
    description: str = Field(
        description="One sentence describing the approach"
    )
    plan_steps: List[str] = Field(
        description="Numbered steps in format '1. Action', '2. Action'."
    )
    first_action: str = Field(
        description="The exact first tool call to execute"
    )
    risks: List[str] = Field(
        description="Top 1-3 risks"
    )
    success_criteria: List[str] = Field(
        description="1-3 observable criteria that indicate task is done"
    )

    model_config = {"extra": "forbid"}


class PlanRanking(NullSafeListModel):
    reasoning: str = Field(
        description="Brief rationale for ranking (1-2 sentences)"
    )
    tool: Literal["plan_review"] = Field(
        description="Constant discriminator; must be 'plan_review'"
    )
    ranking: List[str] = Field(
        description="Plan labels best-to-worst"
    )
    top_choice: str = Field(
        description="Best plan label"
    )
    concerns: List[str] = Field(
        default_factory=list, description="Top concerns"
    )

    model_config = {"extra": "forbid"}


class ChairmanPlan(NullSafeListModel):
    tool: Literal["chairman_plan"] = Field(
        description="Constant discriminator; must be 'chairman_plan'"
    )
    adopted_label: str = Field(
        description="Adopted plan label"
    )
    merged_plan: List[str] = Field(
        description="Numbered steps: '1. Action', '2. Action'."
    )
    watchouts: List[str] = Field(
        default_factory=list, description="Top risks"
    )
    first_step: str = Field(
        description="First tool call to execute"
    )

    model_config = {"extra": "forbid"}


class ReportTaskCompletion(NullSafeListModel):
    tool: Literal["report_completion"] = Field(
        default="report_completion",
        description="Constant discriminator; must be 'report_completion'"
    )
    completed_steps_laconic: List[str] = Field(
        description="Brief bullet list of what was done to finish"
    )
    code: Literal["completed", "failed"] = Field(
        description="Signal whether the task is completed successfully or failed"
    )

    model_config = {"extra": "forbid"}


NEXT_ACTION_FUNCTION = Union[
    ReportTaskCompletion,
    store.Req_ListProducts,
    store.Req_ViewBasket,
    store.Req_ApplyCoupon,
    store.Req_RemoveCoupon,
    store.Req_AddProductToBasket,
    store.Req_RemoveItemFromBasket,
    store.Req_CheckoutBasket,
]


class NextAction(NullSafeListModel):
    tool: Literal["next_action"] = Field(
        description="Constant discriminator; must be 'next_action'"
    )
    intent: str = Field(
        description="What this action achieves (1 sentence)"
    )
    plan_alignment: str = Field(
        description="How it advances the plan (1 sentence)"
    )
    function: NEXT_ACTION_FUNCTION = Field(
        description="Tool call or report_completion"
    )

    model_config = {"extra": "forbid"}


class ActionReview(NullSafeListModel):
    tool: Literal["action_review"] = Field(
        description="Constant discriminator; must be 'action_review'"
    )
    target_label: str = Field(
        description="Best action label"
    )
    approves: bool = Field(
        description="Is target action safe to execute"
    )
    ranking: List[str] = Field(
        description="Action labels best-to-worst"
    )
    concerns: List[str] = Field(
        default_factory=list, description="Top concerns"
    )

    model_config = {"extra": "forbid"}


class ChairmanAction(NullSafeListModel):
    tool: Literal["chairman_action"] = Field(
        description="Constant discriminator; must be 'chairman_action'"
    )
    selected_label: str = Field(
        description="Action label to execute"
    )
    stop: bool = Field(
        description="True only if task should end now"
    )
    updated_plan: List[str] = Field(
        default_factory=list, description="Updated numbered steps if needed"
    )
    rationale: str = Field(
        description="Why this action (1-2 sentences)"
    )

    model_config = {"extra": "forbid"}


class ResultFeedback(NullSafeListModel):
    tool: Literal["result_feedback"] = Field(
        description="Constant discriminator; must be 'result_feedback'"
    )
    task_completed: bool = Field(
        description="Is task done after this result"
    )
    blockers: List[str] = Field(
        default_factory=list, description="Blockers if any"
    )
    plan_adjustments: List[str] = Field(
        default_factory=list, description="Plan changes if needed"
    )
    next_focus: str = Field(
        description="What to do next (1 sentence)"
    )

    model_config = {"extra": "forbid"}


class ChairmanUpdate(NullSafeListModel):
    tool: Literal["chairman_update"] = Field(
        description="Constant discriminator; must be 'chairman_update'"
    )
    stop: bool = Field(
        description="True if task complete or failed"
    )
    updated_plan: List[str] = Field(
        default_factory=list, description="Updated numbered steps if plan changed"
    )
    summary: str = Field(
        description="Brief status and next step (1-2 sentences)"
    )

    model_config = {"extra": "forbid"}


class MemoryNote(NullSafeListModel):
    tool: Literal["memory_note"] = Field(
        description="Constant discriminator; must be 'memory_note'"
    )
    reasoning: str = Field(
        description="What this result means (1 sentence)"
    )
    note: str = Field(
        description="Key facts: IDs, totals, errors (comma-separated)"
    )

    model_config = {"extra": "forbid"}
