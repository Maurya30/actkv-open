"""The vault env must need memory: the code is read once and needed `trail` + 1 observations later."""
import re

from actkv.agent import VaultEnv

CODE = re.compile(r"vault code is ([a-z]+-\d{4})")
START = re.compile(r"trail to the vault starts in room (\d+)")
ARROW = re.compile(r"arrow pointing to room (\d+)")
ROOM = re.compile(r"^Room (\d+) contains")


def play(env, i, memory=None):
    """Scripted agent that only sees its last `memory` observations (None = all of them)."""
    env.reset(i)
    seen = []
    for _ in range(env.max_rounds):
        last = seen[-1] if seen else ""
        if "the vault, with a keypad" in last:
            m = CODE.search(" ".join(seen if memory is None else seen[-memory:]))
            return env.step("submit", {"code": m.group(1) if m else "amber-0000"})[2]
        m = ARROW.search(last) or START.search(last)
        r = ROOM.match(last)
        room = int(m.group(1)) if m else (int(r.group(1)) % env.n_rooms + 1 if r else 1)  # else sweep
        seen.append(env.step("search_room", {"room": f"room {room}"})[0])
    return False


def test_agent_with_memory_solves_every_task():
    env = VaultEnv(n_tasks=40)
    assert all(play(env, i) for i in range(40))


def test_agent_without_memory_fails_every_task():
    env = VaultEnv(n_tasks=40)
    assert not any(play(env, i, memory=1) for i in range(40))


def test_the_code_is_needed_trail_plus_one_observations_later():
    env = VaultEnv(n_tasks=40, trail=4)
    assert not any(play(env, i, memory=4) for i in range(40))
    assert all(play(env, i, memory=5) for i in range(40))


def test_note_reads_once_and_submit_only_works_at_the_vault():
    env = VaultEnv(n_tasks=1)
    env.reset(0)
    room = {"room": f"room {env.note_room}"}
    assert env.code in env.step("search_room", room)[0]
    assert env.code not in env.step("search_room", room)[0]
    assert env.step("submit", {"code": env.code}) == ("There is no vault here.", False, False)


def test_blind_guessing_is_hopeless():
    assert len(VaultEnv.WORDS) * 10_000 >= 100_000
    env = VaultEnv(n_tasks=200)
    codes = set()
    for i in range(200):
        env.reset(i)
        codes.add(env.code)
    assert len(codes) > 190          # tasks don't share a handful of codes
