"""
Universal LLM client wrapper supporting OpenAI, OpenRouter, and Cerebras.
Enforces structured outputs across all providers.
"""

import time
import copy
import random
from typing import List, Dict, Type, Tuple, Any, Optional, Literal, Callable
from openai import OpenAI, RateLimitError, APIError
from pydantic import BaseModel


MAX_RETRIES = 10
BASE_DELAY = 1.0
MAX_DELAY = 60.0


def retry_with_backoff(func: Callable, max_retries: int = MAX_RETRIES) -> Any:
    """
    Execute function with exponential backoff retry on rate limit errors.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                print(f"[RETRY] Rate limit hit, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise
        except APIError as e:
            if e.status_code and e.status_code >= 500:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    print(f"[RETRY] Server error {e.status_code}, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
            else:
                raise

    raise last_exception

_cerebras_adapter = None

def _get_cerebras_adapter():
    global _cerebras_adapter
    if _cerebras_adapter is None:
        from . import cerebras_adapter
        _cerebras_adapter = cerebras_adapter
    return _cerebras_adapter


ProviderType = Literal["openai", "openrouter", "cerebras"]


def _clean_schema_for_cerebras(schema: dict) -> dict:
    """
    Recursively clean schema for Cerebras compatibility:
    - Add 'additionalProperties': false to all objects
    - Remove unsupported constraints: minLength, maxLength, minItems, maxItems
    """
    if not isinstance(schema, dict):
        return schema

    result = copy.deepcopy(schema)

    unsupported_keys = ["minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"]
    for key in unsupported_keys:
        result.pop(key, None)

    if result.get("type") == "object" and "properties" in result:
        result["additionalProperties"] = False

    for key, value in result.get("properties", {}).items():
        result["properties"][key] = _clean_schema_for_cerebras(value)

    if "items" in result:
        result["items"] = _clean_schema_for_cerebras(result["items"])

    for key, value in result.get("$defs", {}).items():
        result["$defs"][key] = _clean_schema_for_cerebras(value)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in result:
            result[key] = [_clean_schema_for_cerebras(item) for item in result[key]]

    return result


def prepare_schema_for_provider(response_format: Type[BaseModel], provider: ProviderType) -> dict:
    """
    Convert Pydantic model to JSON Schema and prepare it for the specific provider.
    """
    schema = response_format.model_json_schema()

    if provider in ("cerebras", "openrouter"):
        schema = _clean_schema_for_cerebras(schema)

    return schema


class StructuredLLMClient:
    """
    Universal structured output client supporting multiple providers.
    """

    def __init__(
        self,
        provider: ProviderType = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self.provider = provider
        self.client = self._build_client(provider, api_key, base_url, default_headers)

    @staticmethod
    def _build_client(
        provider: ProviderType,
        api_key: Optional[str],
        base_url: Optional[str],
        default_headers: Optional[Dict[str, str]],
    ) -> OpenAI:
        resolved_base = base_url or ("https://api.cerebras.ai/v1" if provider == "cerebras" else None)
        return OpenAI(api_key=api_key, base_url=resolved_base, default_headers=default_headers)

    def structured_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        max_completion_tokens: int = 9192,
    ) -> Tuple[BaseModel, Dict[str, Any], float]:
        """
        Call a model and parse directly into the provided Pydantic schema.
        Returns (parsed, usage, duration_sec).
        """
        started = time.time()

        if self.provider == "openai":
            parsed, usage = self._call_openai(model, messages, response_format, max_completion_tokens)
        elif self.provider == "cerebras":
            parsed, usage = self._call_cerebras(model, messages, response_format, max_completion_tokens)
        else:
            parsed, usage = self._call_openrouter(model, messages, response_format, max_completion_tokens)

        duration = time.time() - started
        return parsed, usage, duration

    def _call_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        max_completion_tokens: int,
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
        )
        parsed = completion.choices[0].message.parsed
        return parsed, completion.usage

    def _call_cerebras(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        max_completion_tokens: int,
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        adapter, actual_schema, actual_messages, adapted = self._prepare_cerebras_request(messages, response_format)
        schema = prepare_schema_for_provider(actual_schema, self.provider)

        completion = retry_with_backoff(
            lambda: self.client.chat.completions.create(
                model=model,
                messages=actual_messages,
                response_format=self._json_schema_payload(actual_schema.__name__, schema),
                max_tokens=max_completion_tokens,
            )
        )

        content = completion.choices[0].message.content
        if adapted:
            print(f"[CEREBRAS] Raw response: {content[:200]}...")

        cerebras_parsed = actual_schema.model_validate_json(content)
        parsed = adapter.convert_cerebras_response(cerebras_parsed, response_format)

        if adapted:
            print(f"[CEREBRAS] Converted to: {type(parsed).__name__}, function: {type(parsed.function).__name__}")

        return parsed, completion.usage

    def _prepare_cerebras_request(
        self,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
    ) -> Tuple[Any, Type[BaseModel], List[Dict[str, str]], bool]:
        adapter = _get_cerebras_adapter()
        actual_schema = adapter.get_cerebras_schema(response_format)
        adapted = actual_schema != response_format
        actual_messages = list(messages)

        if adapted:
            print(f"[CEREBRAS] Schema adapted: {response_format.__name__} -> {actual_schema.__name__}")
            enhanced_prompt = adapter.enhance_prompt_for_cerebras(
                actual_messages[-1]["content"],
                response_format
            )
            actual_messages[-1] = {**actual_messages[-1], "content": enhanced_prompt}

        return adapter, actual_schema, actual_messages, adapted

    def _call_openrouter(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        max_completion_tokens: int,
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        schema = prepare_schema_for_provider(response_format, self.provider)

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=self._json_schema_payload(response_format.__name__, schema),
            max_tokens=max_completion_tokens,
        )

        content = completion.choices[0].message.content
        parsed = response_format.model_validate_json(content)
        return parsed, completion.usage

    @staticmethod
    def _json_schema_payload(schema_name: str, schema: dict) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        }
