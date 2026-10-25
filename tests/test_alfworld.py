"""ALFWorldEnv without alfworld installed: fake TextWorld stack and fake alfworld modules."""
import importlib.util
import os
import sys
import types

import pytest

from actkv.agent.envs import ALFWorldEnv

FILES = ["/data/valid_unseen/c-task/trial_3/game.tw-pddl",
         "/data/valid_unseen/a-task/trial_1/game.tw-pddl",
         "/data/valid_unseen/b-task/trial_2/game.tw-pddl"]


class FakeTW:
    """Stands in for the TextWorld wrapper stack: load(game), then reset() / step(command)."""

    def __init__(self):
        self.loaded = []

    def load(self, game_file):
        self.loaded.append(game_file)

    def reset(self):
        return f"You are in {self.loaded[-1]}.", {"admissible_commands": ["look", "win"], "won": False}

    def step(self, cmd):
        won = cmd == "win"
        return f"did {cmd}", 0, won or cmd == "lose", {"admissible_commands": ["look"], "won": won}


def fake_alfworld(monkeypatch, game_files):
    """Install fake alfworld modules laid out like alfworld>=0.4 (the package exports no classes)."""
    calls = []

    class AlfredTWEnv:
        def __init__(self, config, train_eval):
            calls.append((config, train_eval))
            self.game_files = list(game_files)

    tw_mod = types.ModuleType("alfworld.agents.environment.alfred_tw_env")
    tw_mod.AlfredTWEnv = AlfredTWEnv
    for name in ("alfworld", "alfworld.agents", "alfworld.agents.environment"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, tw_mod.__name__, tw_mod)
    return calls


def test_reset_i_always_loads_game_i():
    tw = FakeTW()
    env = ALFWorldEnv(game_files=FILES, tw_env=tw)
    order = [2, 0, 2, 1, 0]
    for i in order:
        env.reset(i)
    assert tw.loaded == [sorted(FILES)[i] for i in order]
    assert env.game_id == "a-task/trial_1"
    assert len(env) == 3


def test_default_config_reads_the_split_from_alfworld_data(tmp_path, monkeypatch):
    (tmp_path / "json_2.1.1" / "valid_unseen").mkdir(parents=True)
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path))
    calls = fake_alfworld(monkeypatch, FILES)
    env = ALFWorldEnv(tw_env=FakeTW())
    config, split = calls[0]
    assert split == "eval_out_of_distribution"
    assert config["dataset"]["num_eval_games"] == -1      # ALFWorld must not truncate before we sort
    assert env.game_files == sorted(FILES)


def test_missing_data_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path))
    fake_alfworld(monkeypatch, FILES)
    with pytest.raises(FileNotFoundError, match="alfworld-download"):
        ALFWorldEnv(tw_env=FakeTW())


def test_no_games_is_an_error_not_a_hang():
    with pytest.raises(RuntimeError, match="no ALFWorld games"):
        ALFWorldEnv(game_files=[], tw_env=FakeTW())


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split"):
        ALFWorldEnv(split="valid_unseen", tw_env=FakeTW())


def test_step_maps_won_and_done():
    env = ALFWorldEnv(game_files=FILES, tw_env=FakeTW())
    assert "Admissible commands: look, win" in env.reset(0)
    assert env.step("act", {"command": "win"})[1:] == (True, True)
    assert env.step("act", {"command": "lose"})[1:] == (True, False)
    assert env.step("act", {"command": "look"})[1:] == (False, False)
    assert env.step("jump", {})[1:] == (False, False)


DATA = os.path.join(os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
                    "json_2.1.1", "valid_unseen")


@pytest.mark.skipif(importlib.util.find_spec("alfworld") is None or not os.path.isdir(DATA),
                    reason="needs `pip install alfworld` and `alfworld-download`")
def test_real_alfworld_gives_the_same_game_for_the_same_index():
    env = ALFWorldEnv()
    first = env.reset(0)
    env.reset(len(env) - 1)
    assert env.reset(0) == first
