"""ReAct episode loop driving an ActKVSession."""
from __future__ import annotations

import time

from ..hf.session import ActKVSession
from .envs import Env


def run_episode(session: ActKVSession, env: Env, fmt, task_idx: int,
                max_new_tokens: int = 4096, verbose: bool = False) -> dict:
    session.reset()
    task = env.reset(task_idx)
    session.prefill(fmt.initial(env.system, task, env.tools), protected=True)
    t0 = time.time()
    success, rounds, invalid = False, 0, 0
    transcript = []
    for rounds in range(1, env.max_rounds + 1):
        text = session.generate(max_new_tokens=max_new_tokens)
        action = fmt.parse_action(text)
        if action is None:
            invalid += 1
            obs, done = "Invalid output. Call exactly one tool using the <tool_call> format.", False
        else:
            obs, done, success = env.step(action["name"], action["arguments"])
        transcript.append({"model": text, "action": action, "obs": obs[:500]})
        if verbose:
            print(f"--- round {rounds} | cache {session.phys} | seen {session.seen}")
            print(text[-400:])
            print(">>", obs[:200])
        if done:
            session.stats.ora_lens.append(session.round_tokens)
            break
        session.end_round()
        session.prefill(fmt.observation(obs, assistant_closed=session.ended_with_eos))
    st = session.stats
    return {
        "task": task_idx, "success": bool(success), "rounds": rounds, "invalid": invalid,
        "peak_entries": st.peak_entries, "trace_len": st.trace_len, "compressions": st.compressions,
        "budgets": st.budgets, "ora_lens": st.ora_lens, "seconds": time.time() - t0,
        "transcript": transcript,
    }

