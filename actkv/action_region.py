"""Detect action regions in generated text by matching the model's tool-call template.

The paper identifies action boundaries by matching generated tokens against the LLM's
standardized output template (Qwen function calling, OpenAI Harmony). We do the same on the
decoded text stream, which works whether the markers are single special tokens or not.
"""
from __future__ import annotations

from dataclasses import dataclass

TEMPLATES = {
    "qwen": ("<tool_call>", "</tool_call>"),
    "harmony": ("<|channel|>commentary to=", "<|call|>"),
}


@dataclass
class ActionRegionDetector:
    start: str = "<tool_call>"
    end: str = "</tool_call>"

    def __post_init__(self):
        self.reset()

    @classmethod
    def for_template(cls, name: str) -> "ActionRegionDetector":
        s, e = TEMPLATES[name]
        return cls(s, e)

    def reset(self) -> None:
        self.buf = ""
        self.inside = False
        self.closed = False      # an action region finished during this round

    def feed(self, piece: str) -> tuple[bool, bool]:
        """Feed the decoded text of one generated token.

        Returns (is_action_token, entered). `is_action_token` is True for every token from the one
        completing the start marker through the one completing the end marker. `entered` is True
        on the token that opens a region (so policies can reset their per-region state).
        """
        self.buf += piece
        entered = False
        if not self.inside:
            idx = self.buf.find(self.start)
            if idx < 0:
                self.buf = self.buf[-(len(self.start) - 1):] if len(self.start) > 1 else ""
                return False, False
            self.inside, entered = True, True
            self.buf = self.buf[idx + len(self.start):]
        if self.end in self.buf:
            self.inside = False
            self.closed = True
            self.buf = ""
        return True, entered
