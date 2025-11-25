"""
Cerebras-specific adapter for structured outputs.
Converts Union-based schemas to simple tool_name + tool_args format
to work around Cerebras' 5-type anyOf limit.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Type, List
from pydantic import BaseModel, Field
from typing import Literal

from erc3 import store
from . import schemas


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    request_cls: Type[BaseModel]
    defaults: Dict[str, Any] = field(default_factory=dict)
    example_args: Any = field(default_factory=dict)
    is_completion: bool = False

    def example_json(self) -> str:
        return json.dumps(self.example_args or {})


TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="ListProducts",
        request_cls=store.Req_ListProducts,
        defaults={"offset": 0, "limit": 3},
        example_args={"offset": 0, "limit": 3},
    ),
    ToolDefinition(name="ViewBasket", request_cls=store.Req_ViewBasket),
    ToolDefinition(
        name="AddProductToBasket",
        request_cls=store.Req_AddProductToBasket,
        example_args={"sku": "gpu-h100", "quantity": 2},
    ),
    ToolDefinition(
        name="RemoveItemFromBasket",
        request_cls=store.Req_RemoveItemFromBasket,
        example_args={"sku": "gpu-h100", "quantity": 1},
    ),
    ToolDefinition(
        name="ApplyCoupon",
        request_cls=store.Req_ApplyCoupon,
        example_args={"coupon": "SAVE10"},
    ),
    ToolDefinition(name="RemoveCoupon", request_cls=store.Req_RemoveCoupon),
    ToolDefinition(name="CheckoutBasket", request_cls=store.Req_CheckoutBasket),
    ToolDefinition(
        name="ReportCompletion",
        request_cls=schemas.ReportTaskCompletion,
        defaults={"completed_steps_laconic": ["Task completed"], "code": "completed"},
        example_args={"completed_steps_laconic": ["step1"], "code": "completed"},
        is_completion=True,
    ),
]

AVAILABLE_TOOL_NAMES = [tool.name for tool in TOOL_DEFINITIONS]
TOOL_BY_NAME = {tool.name.lower(): tool for tool in TOOL_DEFINITIONS}
TOOL_BY_CLASS = {tool.request_cls: tool for tool in TOOL_DEFINITIONS}


class CerebrasNextAction(BaseModel):
    """Simplified NextAction without Union - uses tool_name + tool_args_json."""
    tool: Literal["next_action"] = Field(
        description="Constant discriminator; must be 'next_action'"
    )
    intent: str = Field(
        description="What you are trying to achieve with this single action"
    )
    plan_alignment: str = Field(
        description="How this action advances the current plan"
    )
    tool_name: str = Field(
        description="Tool to call. One of: ListProducts, ViewBasket, AddProductToBasket, RemoveItemFromBasket, ApplyCoupon, RemoveCoupon, CheckoutBasket, ReportCompletion"
    )
    tool_args_json: str = Field(
        default="{}",
        description="JSON string with tool arguments. Examples: {} for no args, {\"sku\":\"gpu-h100\",\"quantity\":2} for AddProductToBasket, {\"coupon\":\"SAVE10\"} for ApplyCoupon"
    )

    model_config = {"extra": "forbid"}


def _parse_tool_args(tool_args_json: str) -> Dict[str, Any]:
    if not tool_args_json:
        return {}
    try:
        return json.loads(tool_args_json)
    except json.JSONDecodeError:
        return {}


def _resolve_tool_definition(tool_name: str) -> ToolDefinition:
    tool = TOOL_BY_NAME.get(tool_name.lower())
    if tool is None:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {AVAILABLE_TOOL_NAMES}")
    return tool


def _build_report_completion(args: Dict[str, Any]) -> schemas.ReportTaskCompletion:
    completed_steps = args.get("completed_steps_laconic", args.get("completed_steps", ["Task completed"]))
    if isinstance(completed_steps, str):
        completed_steps = [completed_steps]
    code = args.get("code", "completed")
    return schemas.ReportTaskCompletion(completed_steps_laconic=completed_steps, code=code)


def _build_tool_function(tool_def: ToolDefinition, tool_args: Dict[str, Any]) -> BaseModel:
    payload = {**tool_def.defaults, **tool_args}
    if tool_def.is_completion:
        return _build_report_completion(payload)

    try:
        return tool_def.request_cls(**payload)
    except Exception as exc:
        raise ValueError(f"Failed to create {tool_def.name} with args {payload}: {exc}")


def _dump_tool_args(function: BaseModel) -> Dict[str, Any]:
    try:
        data = function.model_dump(exclude_none=True)
    except Exception:
        data = {}

    data.pop("tool", None)
    return data


def cerebras_to_next_action(cerebras_action: CerebrasNextAction) -> schemas.NextAction:
    """
    Convert CerebrasNextAction (with tool_name + tool_args_json) to standard NextAction (with Union function).
    """
    tool_def = _resolve_tool_definition(cerebras_action.tool_name)
    tool_args = _parse_tool_args(cerebras_action.tool_args_json)
    function = _build_tool_function(tool_def, tool_args)

    return schemas.NextAction(
        tool="next_action",
        intent=cerebras_action.intent,
        plan_alignment=cerebras_action.plan_alignment,
        function=function
    )


def next_action_to_cerebras(action: schemas.NextAction) -> CerebrasNextAction:
    """
    Convert standard NextAction to CerebrasNextAction format.
    Useful for logging/debugging.
    """
    function = action.function
    tool_def = TOOL_BY_CLASS.get(type(function))
    tool_name = tool_def.name if tool_def else type(function).__name__
    tool_args = _dump_tool_args(function)

    return CerebrasNextAction(
        tool="next_action",
        intent=action.intent,
        plan_alignment=action.plan_alignment,
        tool_name=tool_name,
        tool_args_json=json.dumps(tool_args)
    )


CEREBRAS_SCHEMA_MAP = {
    schemas.NextAction: CerebrasNextAction,
}


def get_cerebras_schema(original_schema: Type[BaseModel]) -> Type[BaseModel]:
    """Get the Cerebras-compatible schema for a given original schema."""
    return CEREBRAS_SCHEMA_MAP.get(original_schema, original_schema)


def convert_cerebras_response(
    cerebras_response: BaseModel,
    original_schema: Type[BaseModel]
) -> BaseModel:
    """Convert a Cerebras response back to the original schema type."""
    if original_schema == schemas.NextAction and isinstance(cerebras_response, CerebrasNextAction):
        return cerebras_to_next_action(cerebras_response)
    return cerebras_response


def _build_tool_description() -> str:
    lines = ["AVAILABLE TOOLS (use exact tool_name values):"]
    for tool in TOOL_DEFINITIONS:
        args_example = json.dumps(tool.example_json())
        lines.append(f"- {tool.name}: tool_args_json: {args_example}")
    lines.append("CRITICAL: tool_args_json MUST be a valid JSON STRING (with escaped quotes inside).")
    return "\n".join(lines)

CEREBRAS_TOOL_DESCRIPTION = _build_tool_description()


def enhance_prompt_for_cerebras(prompt: str, schema: Type[BaseModel]) -> str:
    """Add Cerebras-specific instructions to prompts when using adapted schemas."""
    if schema == schemas.NextAction:
        return prompt + "\n\n" + CEREBRAS_TOOL_DESCRIPTION
    return prompt
