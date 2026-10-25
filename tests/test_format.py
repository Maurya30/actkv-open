import pytest

from actkv.agent import QwenFormat

# Qwen3's chat template (tokenizer_config.json of Qwen/Qwen3-8B) ends EVERY generation prompt with an
# empty think block when enable_thinking=False, and with a bare assistant header otherwise.
NO_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOL_TURN = "<|im_end|>\n<|im_start|>user\n<tool_response>\nroom 3\n</tool_response><|im_end|>\n"


def test_no_thinking_applies_to_every_turn():
    assert QwenFormat(None, enable_thinking=False).observation("room 3", False) == TOOL_TURN + NO_THINK


def test_thinking_turn_matches_the_template():
    assert QwenFormat(None).observation("room 3", False) == TOOL_TURN + "<|im_start|>assistant\n"


class StrictTok:
    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=True, tokenize=False):
        return "prompt"


def test_unsupported_enable_thinking_is_not_silently_dropped():
    with pytest.raises(TypeError):
        QwenFormat(StrictTok(), enable_thinking=False).initial("sys", "task", [])
