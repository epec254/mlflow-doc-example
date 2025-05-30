"""
custom_prompt_judge  (sync & async, arbitrary choices)
──────────────────────────────────────────────────────
Create evaluator callables that send a prompt to a Databricks‑hosted Claude
endpoint and map the model’s categorical answer to a score.

Two factories are exported:

    • custom_prompt_judge         – synchronous, call directly
    • custom_prompt_judge_async   – asynchronous, `await` the result

Both support **any choice labels** (e.g. 'A', 'B', 'C', 'Yes', '👍'), not
just digits, and require that every `{{placeholder}}` in the prompt template
be provided as a keyword argument when calling the evaluator.
"""

from __future__ import annotations

from mlflow.evaluation import Assessment


import asyncio
import re
from typing import Any, Callable, Dict, Optional, Set
import json

from databricks.sdk import WorkspaceClient

# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Template helpers                                                        │
# ╰──────────────────────────────────────────────────────────────────────────╯

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+?)\}\}")


def _find_placeholders(template: str) -> Set[str]:
    return set(_PLACEHOLDER_RE.findall(template))


def _render_template(template: str, kwargs: Dict[str, Any]) -> str:
    missing = _find_placeholders(template).difference(kwargs.keys())
    if missing:
        raise ValueError(
            f"Missing placeholders {sorted(missing)}; "
            f"kwargs supplied: {list(kwargs.keys())}"
        )

    rendered: str = template
    for ph in _find_placeholders(template):
        rendered = rendered.replace(f"{{{{{ph}}}}}", str(kwargs[ph]))
    return rendered


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Choice‑parsing helper                                                   │
# ╰──────────────────────────────────────────────────────────────────────────╯


def _extract_choice(text: str, valid_choices: Optional[Set[str]]) -> Optional[str]:
    """
    Return the chosen label if it matches one of `valid_choices` (case‑insensitive).

    Priority:
      1. 'FINAL ANSWER: <choice>'  (or 'ANSWER: <choice>')
      2. last standalone occurrence of <choice>
    """
    # Build regex like  (A|B|C|Yes|👍) with proper escaping
    choice_pattern = (
        "|".join(re.escape(c) for c in sorted(valid_choices, key=len, reverse=True))
        if valid_choices
        else ".*"
    )
    tag_re = re.compile(
        rf"(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]?\s*({choice_pattern})", re.I
    )

    tagged = tag_re.search(text)
    if tagged:
        return tagged.group(1)

    # Fallback: last standalone occurrence
    standalone_re = re.compile(rf"\b({choice_pattern})\b", re.I)
    candidates = standalone_re.findall(text)
    return candidates[-1] if candidates else None


# ╭──────────────────────────────────────────────────────────────────────────╮
# │ Core factories                                                          │
# ╰──────────────────────────────────────────────────────────────────────────╯


def _make_evaluator(
    *,
    name: str,
    prompt_template: str,
    choice_values: Optional[Dict[str, Any]] = None,
    model: str,
    temperature: float,
    use_cot: bool,
    async_mode: bool,
) -> Callable[..., Any]:
    """Shared factory used by both sync and async wrappers."""
    w = WorkspaceClient()
    openai_client = w.serving_endpoints.get_open_ai_client()

    # Pre‑compute for speed
    valid_choices = None
    if choice_values:
        valid_choices = set(choice_values.keys())
        canonical_map = {
            c.lower(): c for c in valid_choices
        }  # map for case‑insensitive lookup

    def _build_prompt(kwargs: Dict[str, Any]) -> tuple[str, int]:
        prompt = _render_template(prompt_template, kwargs)
        if not choice_values:
            prompt += (
                "\n\nYou may think step‑by‑step BEFORE giving your answer."
                f"When you are done, write:\n\nFINAL ANSWER: <choice>"
            )
            return prompt, 64
        if use_cot:
            prompt += (
                "\n\nYou may think step‑by‑step BEFORE giving your answer. Your step-by-step reasoning should include the available options listed with their choice identifier."
                f"When you are done, write:\n\nFINAL ANSWER: <{ ' | '.join(valid_choices) }>"
            )
            return prompt, 64
        prompt += f"\n\nAnswer with ONLY one of: {', '.join(valid_choices)}."
        return prompt, 12

    # ────────────────────────────────────────────────────────────────────────
    # synchronous evaluator
    # ────────────────────────────────────────────────────────────────────────
    def _evaluate_sync(**kwargs) -> Dict[str, Any]:
        prompt, max_tokens = _build_prompt(kwargs)

        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # max_tokens=None,
        )
        raw_reply = response.choices[0].message.content.strip()

        choice = _extract_choice(raw_reply, valid_choices)
        if choice is None:
            raise ValueError(
                f"[{name}] Could not find a valid choice in reply:\n{raw_reply}"
            )

        if choice_values:
            canonical_choice = canonical_map[choice.lower()]
            value = choice_values[canonical_choice]
        else:
            value = choice
            # value = canonical_choice

        # return {
        #     "judge_name": name,
        #     "choice": canonical_choice,
        #     "score": score,
        #     "raw_response": raw_reply,
        # }
        return Assessment(
            name=name,
            value=value,
            rationale=(
                raw_reply
                + (
                    "\n\nChoice scores: \n```"
                    + json.dumps(choice_values, indent=2)
                    + "```"
                )
                if choice_values
                else ""
            ),
        )

    # ────────────────────────────────────────────────────────────────────────
    # asynchronous evaluator
    # ────────────────────────────────────────────────────────────────────────
    async def _evaluate_async(**kwargs) -> Dict[str, Any]:
        prompt, max_tokens = _build_prompt(kwargs)

        def _blocking_call():
            return openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = await asyncio.to_thread(_blocking_call)
        raw_reply = response.choices[0].message.content.strip()

        choice = _extract_choice(raw_reply, valid_choices)
        if choice is None:
            raise ValueError(
                f"[{name}] Could not find a valid choice in reply:\n{raw_reply}"
            )

        canonical_choice = canonical_map[choice.lower()]
        score = choice_values[canonical_choice]

        return {
            "judge_name": name,
            "choice": canonical_choice,
            "score": score,
            "raw_response": raw_reply,
        }

    return _evaluate_async if async_mode else _evaluate_sync


# public sync factory
# Fully vibe coded :)
def custom_prompt_judge(
    assessment_name: str,
    prompt_template: str,
    choice_values: Optional[Dict[str, Any]] = None,
    *,
    model: str = "databricks-claude-3-7-sonnet",
    temperature: float = 0.0,
    use_cot: bool = True,
) -> Callable[..., Dict[str, Any]]:
    """Return a **synchronous** evaluator callable."""
    return _make_evaluator(
        name=assessment_name,
        prompt_template=prompt_template,
        choice_values=choice_values,
        model=model,
        temperature=temperature,
        use_cot=use_cot,
        async_mode=False,
    )


# public async factory
def custom_prompt_judge_async(
    assessment_name: str,
    prompt_template: str,
    choice_values: Dict[str, Any],
    *,
    model: str = "databricks-claude-3-7-sonnet",
    temperature: float = 0.0,
    use_cot: bool = True,
) -> Callable[..., "asyncio.Future[Dict[str, Any]]"]:
    """Return an **asynchronous** evaluator callable (`await` it)."""
    return _make_evaluator(
        name=assessment_name,
        prompt_template=prompt_template,
        choice_values=choice_values,
        model=model,
        temperature=temperature,
        use_cot=use_cot,
        async_mode=True,
    )
