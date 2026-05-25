from dataclasses import dataclass
import os
from typing import Iterable, Optional
from dotenv import load_dotenv
from openai import OpenAI

from lgrl_final.subgoal_manager import SUBGOALS

load_dotenv()


def _resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key:
        return api_key
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


def _normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    normalized = []
    prev_underscore = False
    for ch in str(text).upper():
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            normalized.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                normalized.append("_")
                prev_underscore = True
    return "".join(normalized).strip("_")


def _extract_allowed_token(text: str, allowed_tokens: Optional[Iterable[str]]) -> Optional[str]:
    if not text or not allowed_tokens:
        return None

    normalized = _normalize_for_match(text)
    allowed_list = [token for token in allowed_tokens if token]
    if not allowed_list:
        return None

    for token in allowed_list:
        token_norm = _normalize_for_match(token)
        if normalized == token_norm:
            return token

    for token in allowed_list:
        token_norm = _normalize_for_match(token)
        if token_norm and token_norm in normalized:
            return token

    return None


def _extract_message_text(message) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                text = getattr(part, "text", None)
                if text is not None:
                    parts.append(str(text))
        content = "".join(parts)

    if content is None or (isinstance(content, str) and not content.strip()):
        for attr in ("reasoning_content", "reasoning", "thinking"):
            value = getattr(message, attr, None)
            if value:
                content = value
                break

    if content is None:
        return ""

    return str(content)


def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            return {}
    if hasattr(usage, "dict"):
        try:
            return usage.dict()
        except Exception:
            return {}

    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def _merge_usage(primary: dict, secondary: dict) -> dict:
    if not primary:
        return dict(secondary) if secondary else {}
    if not secondary:
        return dict(primary)

    merged = dict(primary)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if key in secondary:
            primary_val = merged.get(key, 0) or 0
            secondary_val = secondary.get(key, 0) or 0
            try:
                merged[key] = int(primary_val) + int(secondary_val)
            except (TypeError, ValueError):
                merged[key] = secondary_val
    return merged

@dataclass
class OpenAICompatibleResponse:
    content: str
    usage: dict


class OpenAICompatibleChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 16,
        timeout: float = 30.0,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict] = None,
        allowed_tokens: Optional[Iterable[str]] = None,
        strict_output: bool = True,
        retry_on_empty: bool = True,
        fallback_token: Optional[str] = None,
        retry_attempts: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.allowed_tokens = list(allowed_tokens) if allowed_tokens else None
        self.strict_output = strict_output
        self.retry_on_empty = retry_on_empty
        self.fallback_token = fallback_token
        self.retry_attempts = max(0, int(retry_attempts))
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _build_system_prompt(self, override: Optional[str] = None) -> str:
        if override is not None:
            return override
        return (
            "Return only the single best subgoal token from the allowed list. "
            "Do not add any explanation."
        )

    def _request(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, stop: Optional[list[str]] = None):
        request_kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(system_prompt),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body
        if stop:
            request_kwargs["stop"] = stop

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = ""
        if getattr(response, "choices", None):
            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            content = _extract_message_text(message)
            if not content:
                content = getattr(first_choice, "text", "") or ""

        usage = _usage_to_dict(getattr(response, "usage", None))
        return content, usage

    def __call__(self, prompt: str):
        return self.invoke(prompt)

    def generate(self, prompt: str):
        return self.invoke(prompt)

    def complete(self, prompt: str):
        return self.invoke(prompt)

    def invoke(self, prompt: str):
        content, usage = self._request(prompt)

        if self.strict_output and self.allowed_tokens:
            token = _extract_allowed_token(content, self.allowed_tokens)
            if token is None and self.retry_on_empty and self.retry_attempts > 0:
                allowed_list = ", ".join(self.allowed_tokens)
                for attempt in range(self.retry_attempts):
                    retry_system = (
                        "Return exactly one token from this list: "
                        f"{allowed_list}. No other text."
                    )
                    if attempt == self.retry_attempts - 1:
                        retry_system += " If unsure, pick the closest match."
                    retry_content, retry_usage = self._request(
                        prompt,
                        system_prompt=retry_system,
                        max_tokens=min(4, self.max_tokens),
                        temperature=0.0,
                    )
                    usage = _merge_usage(usage, retry_usage)
                    token = _extract_allowed_token(retry_content, self.allowed_tokens)
                    if token is not None:
                        break
            if token is None and self.fallback_token:
                token = _extract_allowed_token(self.fallback_token, self.allowed_tokens)
            content = token or ""

        return OpenAICompatibleResponse(content=content, usage=usage)


def build_llm_client(
    provider: Optional[str],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16,
    timeout: float = 30.0,
    allowed_tokens: Optional[Iterable[str]] = None,
    strict_output: bool = True,
    retry_on_empty: bool = True,
    fallback_token: Optional[str] = None,
    retry_attempts: int = 2,
):
    if provider is None or provider == "none":
        return None

    normalized_provider = provider.strip().lower()
    if normalized_provider != "deepseek":
        raise ValueError(f"Unsupported LLM provider: {provider}")

    resolved_api_key = _resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError(
            "DeepSeek provider selected but no API key was provided. Set --llm-api-key or DEEPSEEK_API_KEY."
        )

    resolved_base_url = base_url or "https://api.deepseek.com"
    resolved_model = model or "deepseek-v4-pro"
    reasoning_effort = "high" if "pro" in resolved_model.lower() else None
    extra_body = {"thinking": {"type": "enabled"}} if reasoning_effort is not None else None

    effective_allowed_tokens = list(allowed_tokens) if allowed_tokens is not None else list(SUBGOALS)

    return OpenAICompatibleChatClient(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        allowed_tokens=effective_allowed_tokens,
        strict_output=strict_output,
        retry_on_empty=retry_on_empty,
        fallback_token=fallback_token,
        retry_attempts=retry_attempts,
    )