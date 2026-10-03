"""Incremental chat formatting.

The cache is append-only, so we never re-render the whole conversation (Qwen3's template also
strips <think> blocks from history, which would desync the cache). The first prompt goes through
the tokenizer's chat template; later turns append only the new tool-response segment.
"""
from __future__ import annotations

import json
import re


class QwenFormat:
    """Qwen2.5 / Qwen3 function-calling format (Hermes-style <tool_call> JSON)."""

    template = "qwen"
    tool_call_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

    def __init__(self, tokenizer, enable_thinking: bool = True):
        self.tok = tokenizer
        self.enable_thinking = enable_thinking

    def initial(self, system: str, task: str, tools: list[dict]) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        kw = {"enable_thinking": self.enable_thinking}
        try:
            return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                tokenize=False, **kw)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                tokenize=False)

    def observation(self, obs: str, assistant_closed: bool) -> str:
        close = "" if assistant_closed else "<|im_end|>"
        return (f"{close}\n<|im_start|>user\n<tool_response>\n{obs}\n</tool_response>"
                f"<|im_end|>\n<|im_start|>assistant\n")

    def parse_action(self, text: str) -> dict | None:
        m = self.tool_call_re.findall(text)
        if not m:
            return None
        try:
            call = json.loads(m[-1])
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call:
            return None
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"name": call["name"], "arguments": args if isinstance(args, dict) else {}}


def tool(name: str, description: str, **params: str) -> dict:
    """Small helper to declare an OpenAI-style function schema with string params."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                       "required": list(params)}}}
