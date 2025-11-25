"""
Prompt helpers for council stages.
"""

from typing import List, Dict


AGENT_SYSTEM_PROMPT = """
<ROLE>
You are an expert agent in a council of LLMs solving e-commerce benchmark tasks with tools.
</ROLE>

<BREVITY>
- Keep responses concise: 1-2 sentences per field
- Format plan_steps as numbered list: "1. Action", "2. Action", etc.
</BREVITY>

<AVAILABLE_TOOLS>
- ListProducts(offset, limit): Get catalog of available products (returns: sku, name, price, available)
- ViewBasket: See current basket state (returns: items, subtotal, discount, total)
- AddProductToBasket(sku, quantity): Add item to basket
- RemoveItemFromBasket(sku, quantity): Remove item from basket
- ApplyCoupon(coupon): Apply discount coupon to basket
- RemoveCoupon: Remove applied coupon
- CheckoutBasket: Complete the purchase (finalizes order)
- ReportTaskCompletion(completed_steps, code): Signal task completion or failure
</AVAILABLE_TOOLS>

<KEY_RULES>
- Use EXACT IDs from ListProducts results
- "Buy ALL X" means buy ALL available units (quantity = available field value)
- Verify basket state before checkout
</KEY_RULES>
"""

CHAIRMAN_SYSTEM_PROMPT = """
<ROLE>
You are the chairman of an LLM council. You make final decisions on actions and plan updates.
All proposals and reviews are anonymized - judge them by content quality only.
</ROLE>

<DECISION_PRINCIPLES>
1. Be decisive: prefer the simplest safe action that advances the task
2. Trust consensus: if multiple reviewers agree, weight their opinion heavily
3. Verify completion: only stop when success criteria are clearly met
4. Format plan steps as numbered list: "1. Action", "2. Action"
</DECISION_PRINCIPLES>

<OUTPUT_REQUIREMENTS>
- Return only the requested structured object
- Set stop=true only when task is definitively complete or failed
- Keep updated_plan minimal; avoid unnecessary rewrites
</OUTPUT_REQUIREMENTS>
"""

MEMORY_SYSTEM_PROMPT = """
You are a memory assistant that compresses tool call results into concise, factual notes.
- Capture key identifiers, quantities, statuses, and any errors.
- Keep the note compact (1-2 sentences max).
"""


def format_plan_prompt(task_text: str) -> str:
    return f"""Draft a short plan for the task below. Focus on concrete tool-using steps.

Task:
{task_text}

IMPORTANT:
- Format plan_steps as numbered list: "1. ListProducts to find items", "2. AddProductToBasket", etc.
- "Buy ALL X" means buy ALL available units (use quantity = available field from ListProducts)
- Keep fields concise (1-2 sentences)"""


def format_plan_review_prompt(task_text: str, proposals: List[Dict[str, str]]) -> str:
    proposals_block = "\n\n".join([f"Response {p['label']}:\n{p['summary']}" for p in proposals])
    return f"""You are evaluating different plans for the same task.

Task: {task_text}

Plans (anonymized):
{proposals_block}

Your job:
1) Briefly assess each plan (strengths/weaknesses)
2) Populate structured fields:
- `ranking`: list of Response labels in order best→worst
- `top_choice`: the single best Response label
- `reasoning`: concise rationale (1-2 sentences)"""


def format_chairman_plan_prompt(task_text: str, proposals: List[Dict[str, str]], reviews: List[Dict[str, str]]) -> str:
    proposals_block = "\n\n".join(
        [f"{p['label']}: {p['summary']}" for p in proposals]
    )
    reviews_block = "\n\n".join(
        [f"Reviewer {idx + 1}: ranked={', '.join(r['ranking'])}, prefers={r['top_choice']}"
         for idx, r in enumerate(reviews)]
    )
    return f"""You must pick or merge the best plan using peer feedback.

Task:
{task_text}

Candidate plans (anonymized):
{proposals_block}

Peer rankings (anonymized):
{reviews_block}

Adopt or merge the best plan. Format merged_plan as numbered list: "1. Action", "2. Action", etc."""


def format_action_prompt(task_text: str, plan: List[str], current_step_index: int, history: str) -> str:
    plan_block = "\n".join([f"- {step}" for step in plan]) if plan else "No plan yet."
    current_step = plan[current_step_index] if plan and current_step_index < len(plan) else "No current step."
    step_number = current_step_index + 1
    return f"""Propose the single next tool call to execute the current step of the plan.

Task:
{task_text}

Current plan:
{plan_block}

CURRENT STEP ({step_number}/{len(plan)}): {current_step}

Your goal: propose a tool call that accomplishes the current step above.

History:
{history}

Fill the NextAction schema: set `intent`, `plan_alignment`, and select `function` (tool call or report_completion)."""


def format_action_review_prompt(task_text: str, proposals: List[Dict[str, str]]) -> str:
    proposals_block = "\n\n".join(
        [f"{p['label']}: {p['summary']}" for p in proposals]
    )
    return f"""Review and rank the proposed actions (anonymized). Approve only actions that are safe and relevant.

Task:
{task_text}

Proposed actions:
{proposals_block}

Pick the best option and flag any risks.
Set `target_label` to the best proposal label, `approves` to indicate safety, and fill `ranking` best-to-worst."""


def format_result_feedback_prompt(task_text: str, last_result: str, history: str, plan: List[str]) -> str:
    plan_block = "\n".join([f"- {s}" for s in plan]) if plan else "No plan yet."
    return f"""Assess the latest tool result and state whether the task is done.

Task:
{task_text}

Plan:
{plan_block}

Recent result:
{last_result}

History:
{history}

Populate `task_completed`, `next_focus`, and any `blockers`."""


def format_chairman_action_prompt(task_text: str, proposals: List[Dict[str, str]], reviews: List[Dict[str, str]], plan: List[str], history: str) -> str:
    proposals_block = "\n\n".join(
        [f"{p['label']}: {p['summary']}" for p in proposals]
    )
    reviews_block = "\n\n".join(
        [f"Reviewer {idx + 1}: ranking={', '.join(r['ranking'])}, target={r['target']}, approves={r['approves']}"
         for idx, r in enumerate(reviews)]
    )
    plan_block = "\n".join([f"- {s}" for s in plan]) if plan else "No plan yet."
    return f"""Select the action to execute now, using peer reviews and the current plan.

Task:
{task_text}

Plan:
{plan_block}

History:
{history}

Proposals (anonymized):
{proposals_block}

Peer Reviews (anonymized):
{reviews_block}

Pick the safest useful action. Set `selected_label` to the action label and `stop` to True only if task should end."""


def format_chairman_result_prompt(task_text: str, result: str, peer_feedback: List[Dict[str, str]], plan: List[str], history: str) -> str:
    feedback_block = "\n\n".join(
        [f"Peer {idx + 1}: done={fb['done']}, next_focus={fb['next_focus']}, blockers={fb.get('blockers', [])}"
         for idx, fb in enumerate(peer_feedback)]
    )
    plan_block = "\n".join([f"- {s}" for s in plan]) if plan else "No plan yet."
    return f"""Update the shared plan after the latest tool result and decide whether to continue.

Task:
{task_text}

Plan:
{plan_block}

History:
{history}

Latest result:
{result}

Peer feedback (anonymized):
{feedback_block}

If the task is complete, set stop=true; otherwise, adjust the plan for the next step."""


def format_memory_prompt(task_text: str, action_summary: str, raw_result: str) -> str:
    return f"""Compress the latest tool call into a concise note (1-2 sentences).

Task:
{task_text}

Action summary:
{action_summary}

Raw tool result:
{raw_result}

Return a short reasoning and factual note (include ids, names, totals, errors)."""
