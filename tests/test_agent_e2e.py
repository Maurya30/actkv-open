import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from actkv.agent import Env, QwenFormat, VaultEnv, run_episode
from actkv.analysis import AAVRecorder
from actkv.budget import BudgetConfig, ConfidenceBudget, FixedBudget
from actkv.eviction import make_policy
from actkv.hf import ActKVSession, SamplingParams


class CharTok:
    """Offline stand-in for a HF tokenizer: one token per byte (mod vocab)."""
    eos_token_id = 0
    V = 256

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([[b % (self.V - 1) + 1 for b in text.encode()]])
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i - 1 for i in ids if i > 0).decode(errors="replace")

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + json.dumps(tools)[:200] + "<|im_start|>assistant\n"


def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=8192, attn_implementation="eager")
    return Qwen3ForCausalLM(cfg).eval()


@pytest.mark.parametrize("method,budget", [
    ("full", None),
    ("actkv", "adaptive"),
    ("actkv", 64),
    ("snapkv", 64),
    ("streaming", 64),
])
def test_episode_runs_and_bounds_memory(method, budget):
    model, tok = tiny(), CharTok()
    fmt = QwenFormat(tok)
    env = VaultEnv(n_tasks=1, n_rooms=3, filler=8)
    env.max_rounds = 4
    if budget == "adaptive":
        b = ConfidenceBudget(BudgetConfig(init_budget=64, window=8, stride=2))
    else:
        b = FixedBudget(budget) if budget else None
    pol = make_policy(method)
    s = ActKVSession(model, tok, pol, budget=b, sampling=SamplingParams(temperature=1.0),
                     mid_round_slack=32, seed=0)
    # random weights never close a <tool_call>, so make the "action region" the whole round
    s.detector.start, s.detector.end = "", "\x00never\x00"
    res = run_episode(s, env, fmt, 0, max_new_tokens=24)
    assert res["rounds"] == 4 and res["invalid"] == 4
    assert res["trace_len"] == s.seen
    if method == "full":
        assert res["peak_entries"] == res["trace_len"] and res["compressions"] == 0
    else:
        assert res["compressions"] > 0
        assert s.phys < s.seen          # memory actually bounded
        assert res["peak_entries"] < res["trace_len"]


def test_scripted_agent_solves_vault():
    env = VaultEnv(n_tasks=3, n_rooms=5, filler=5, seed=1)
    fmt = QwenFormat(CharTok())
    for i in range(3):
        env.reset(i)
        found = None
        for r in range(1, 6):
            obs, done, _ = env.step("search_room", {"room": f"room {r}"})
            if "note" in obs:
                found = (r, obs.split("drawer ")[1][0])
        assert found
        text = ('<think>ok</think><tool_call>{"name": "open_drawer", "arguments": '
                f'{{"room": "room {found[0]}", "drawer": "{found[1]}"}}}}</tool_call>')
        act = fmt.parse_action(text)
        obs, _, _ = env.step(act["name"], act["arguments"])
        code = obs.split("'")[1]
        _, done, ok = env.step("submit", {"code": code})
        assert done and ok


class StepEnv(Env):
    """`finish` ends the episode successfully; any other tool is a no-op."""
    max_rounds = 3

    def reset(self, i):
        return "Call finish."

    def step(self, name, args):
        return "ok", name == "finish", name == "finish"


def call(name):
    return f'<tool_call>{{"name": "{name}", "arguments": {{}}}}</tool_call>'


def script(monkeypatch, tok, rounds):
    """Make the session 'sample' exactly these tokens; a round given as (text, True) ends with EOS."""
    ids = []
    for r in rounds:
        text, eos = r if isinstance(r, tuple) else (r, False)
        ids += tok(text).input_ids[0].tolist() + ([tok.eos_token_id] if eos else [])
    it = iter(ids)
    monkeypatch.setattr("actkv.hf.session.sample", lambda logits, sp, gen=None: next(it))


def aav_episode(monkeypatch, rounds, max_rounds):
    tok = CharTok()
    pol = AAVRecorder()
    s = ActKVSession(tiny(), tok, pol, budget=FixedBudget(10**9))
    script(monkeypatch, tok, rounds)
    env = StepEnv()
    env.max_rounds = max_rounds
    return pol, run_episode(s, env, QwenFormat(tok), 0)


def test_aav_records_the_final_action_round(monkeypatch):
    pol, res = aav_episode(monkeypatch, [call("finish")], max_rounds=3)
    assert res["success"] and res["rounds"] == 1
    assert len(pol.rounds) == 1


def test_aav_skips_a_round_without_action(monkeypatch):
    pol, res = aav_episode(monkeypatch, [call("look"), ("no tool call", True)], max_rounds=2)
    assert res["rounds"] == 2 and res["invalid"] == 1
    assert len(pol.rounds) == 1


def test_sampled_end_token_is_not_cached_and_im_end_is_written(monkeypatch):
    """Qwen3 has two end tokens; only <|im_end|> closes a turn, so we always write it ourselves."""
    tok = CharTok()
    s = ActKVSession(tiny(), tok, make_policy("full"))
    forwarded, prefilled = [], []
    real_forward, real_prefill = s._forward, s.prefill
    monkeypatch.setattr(s, "_forward", lambda ids, want_attn: forwarded.extend(ids.tolist()) or real_forward(ids, want_attn))
    monkeypatch.setattr(s, "prefill", lambda text, **k: prefilled.append(text) or real_prefill(text, **k))
    script(monkeypatch, tok, [("no tool call", True), ("still none", True)])
    env = StepEnv()
    env.max_rounds = 2
    run_episode(s, env, QwenFormat(tok), 0)
    assert tok.eos_token_id not in forwarded
    assert prefilled[1].startswith("<|im_end|>")
