#!/usr/bin/env python3
"""Simple LLM smoke test for DeepSeek/OpenAI-compatible endpoints."""

import argparse
import json
from typing import Any, Optional

from lgrl_final.llm_clients import build_llm_client
from lgrl_final.subgoal_manager import LLMSubgoalManager, SUBGOALS


def _normalize_usage(usage: Any) -> Optional[dict]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            return None
    if hasattr(usage, "dict"):
        try:
            return usage.dict()
        except Exception:
            return None

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _build_prompt() -> str:
    return (
        "Choose exactly one next pentest subgoal from: "
        "DISCOVER_HOST, ENUM_SERVICE, EXPLOIT_ACCESS, PRIV_ESC. "
        "Reply with only one token from the list without outputting anythn.\n"
        "Current subgoal: EXPLOIT_ACCESS\n"
        "Context:\n"
        "- Discovered 2 hosts\n"
        "- Gained user shell on host_1\n"
    )


def _extract_fallback_token(prompt: str) -> Optional[str]:
    for line in prompt.splitlines():
        if line.lower().startswith("current subgoal:"):
            token = line.split(":", 1)[1].strip()
            return token or None
    return None


def _parse_subgoal(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = text.upper()
    for subgoal in SUBGOALS:
        if subgoal in normalized:
            return subgoal
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM smoke test for lgrl_final.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek"], help="LLM provider.")
    parser.add_argument("--api-key", help="API key (defaults to DEEPSEEK_API_KEY or OPENAI_API_KEY).")
    parser.add_argument("--base-url", help="Override base URL.")
    parser.add_argument("--model", default="deepseek-v4-pro", help="Model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max tokens to request.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    parser.add_argument("--prompt", help="Override the default prompt.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of calls to make.")
    parser.add_argument("--fallback-token", help="Token to use when strict output is empty.")
    parser.add_argument("--json", action="store_true", help="Print JSON output only.")
    args = parser.parse_args()

    prompt = args.prompt or _build_prompt()
    fallback_token = args.fallback_token or _extract_fallback_token(prompt) or SUBGOALS[0]

    client = build_llm_client(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        fallback_token=fallback_token,
    )

    if client is None:
        raise SystemExit("No LLM client configured.")

    total_tokens = 0
    calls = max(1, int(args.repeat))
    outputs = []

    for idx in range(calls):
        response = client.invoke(prompt)
        token_usage = LLMSubgoalManager._extract_token_usage(response)
        total_tokens += token_usage
        usage_dict = _normalize_usage(getattr(response, "usage", None))
        content = getattr(response, "content", "") or ""
        parsed_subgoal = _parse_subgoal(content)
        strict_output = content.strip().upper() in SUBGOALS

        outputs.append(
            {
                "call": idx + 1,
                "content": content.strip(),
                "parsed_subgoal": parsed_subgoal,
                "strict_output": strict_output,
                "token_usage": token_usage,
                "usage": usage_dict,
            }
        )

    avg_tokens = float(total_tokens / calls)

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "calls": calls,
                    "average_tokens": avg_tokens,
                    "results": outputs,
                },
                indent=2,
            )
        )
        return 0

    print("LLM Call Test")
    print(f"Model: {args.model}")
    print(f"Calls: {calls}")
    print(f"Fallback token: {fallback_token}")
    for item in outputs:
        print("-")
        print(f"  Call: {item['call']}")
        print(f"  Content: {item['content']}")
        print(f"  Parsed subgoal: {item['parsed_subgoal']}")
        print(f"  Strict output: {item['strict_output']}")
        print(f"  Token usage: {item['token_usage']}")
        if item["usage"] is not None:
            print(f"  Raw usage: {item['usage']}")
    print(f"Average tokens per call: {avg_tokens:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
