"""Save a tiny random Qwen3 model and a byte-level tokenizer, for dry runs of the eval pipeline.

    python scripts/make_tiny_model.py /tmp/tiny     # then: run_eval.py --model /tmp/tiny --dtype float32

The tokenizer has Qwen's special tokens and a minimal Qwen-style chat template (tools, enable_thinking),
so run_eval exercises the same code paths as with a real checkpoint. Outputs are random, of course.
"""
from __future__ import annotations

import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

TEMPLATE = (   # HF renders with trim_blocks, which drops a newline right after a tag: emit those explicitly
    "{% if tools %}<|im_start|>system\n{{ messages[0].content }}\n\n# Tools\n<tools>"
    "{% for t in tools %}{{ '\n' }}{{ t | tojson }}{% endfor %}{{ '\n' }}</tools><|im_end|>\n"
    "{% set rest = messages[1:] %}{% else %}{% set rest = messages %}{% endif %}"
    "{% for m in rest %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


def main(out: str) -> None:
    vocab = {ch: i for i, ch in enumerate(pre_tokenizers.ByteLevel.alphabet())}   # one token per byte
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    tok.add_tokens(["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(out)

    torch.manual_seed(0)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=65536,
                      eos_token_id=eos[0], pad_token_id=eos[1], bos_token_id=None)
    model = Qwen3ForCausalLM(cfg)
    model.generation_config.eos_token_id = eos
    model.save_pretrained(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
