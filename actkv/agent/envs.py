"""Agent environments with a common interface.

    env.system, env.tools          prompt + tool schemas
    env.reset(i) -> task text       start task i
    env.step(name, args) -> (observation, done, success)
"""
from __future__ import annotations

import copy
import os
import random

from .format import tool


class Env:
    system = "You are a helpful agent. Use the tools to solve the task. Call exactly one tool per turn."
    tools: list[dict] = []
    max_rounds = 30

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self, i: int) -> str:
        raise NotImplementedError

    def step(self, name: str, args: dict) -> tuple[str, bool, bool]:
        raise NotImplementedError


class VaultEnv(Env):
    """Offline toy task with long, noisy observations (useful for smoke tests and quick ablations).

    A code word is hidden in one of N rooms. Each room returns a long inventory listing; the room
    with the code also points to the drawer it is in. The agent must `search_room`, then
    `open_drawer`, then `submit` the code. Task-critical facts appear early and are needed many
    rounds later, which is exactly what action-guided retention should keep.
    """

    system = ("You are an agent exploring a building. Use the tools to find the code word and submit it. "
              "Call exactly one tool per turn.")
    tools = [
        tool("search_room", "List the contents of a room.", room="room name, e.g. 'room 3'"),
        tool("open_drawer", "Open a drawer in a room.", room="room name", drawer="drawer label, e.g. 'B'"),
        tool("submit", "Submit the code word.", code="the code word"),
    ]
    WORDS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo",
             "juniper", "krypton", "lagoon", "marble", "nebula", "obsidian", "prairie", "quartz"]
    ITEMS = ["chair", "lamp", "box", "rug", "book", "vase", "clock", "plant", "mug", "poster",
             "cable", "jar", "tray", "fan", "basket", "mirror", "kettle", "towel"]

    def __init__(self, n_tasks: int = 50, n_rooms: int = 8, filler: int = 40, seed: int = 0):
        self.n_tasks, self.n_rooms, self.filler, self.seed = n_tasks, n_rooms, filler, seed
        self.max_rounds = n_rooms + 6

    def __len__(self):
        return self.n_tasks

    def reset(self, i: int) -> str:
        self.rng = random.Random(self.seed * 100_003 + i)
        self.room = self.rng.randrange(1, self.n_rooms + 1)
        self.drawer = self.rng.choice("ABCDEF")
        self.code = self.rng.choice(self.WORDS)
        self.opened = False
        return (f"The building has rooms 1 to {self.n_rooms}. One room contains a note saying which "
                f"drawer holds a code word. Find the code word and submit it.")

    def _room_text(self, r: int) -> str:
        rng = random.Random(self.seed * 7 + r * 131 + self.room)
        items = [f"{rng.choice(self.ITEMS)} #{rng.randrange(1000)}" for _ in range(self.filler)]
        if r == self.room:
            items.insert(rng.randrange(len(items)), f"a note: 'the code is in drawer {self.drawer}'")
        drawers = ", ".join(f"drawer {d}" for d in "ABCDEF")
        return f"Room {r} contains: " + "; ".join(items) + f". Drawers: {drawers}."

    def step(self, name, args):
        try:
            if name == "search_room":
                r = int("".join(ch for ch in str(args.get("room", "")) if ch.isdigit()) or -1)
                if not 1 <= r <= self.n_rooms:
                    return f"There is no {args.get('room')}.", False, False
                return self._room_text(r), False, False
            if name == "open_drawer":
                r = int("".join(ch for ch in str(args.get("room", "")) if ch.isdigit()) or -1)
                d = str(args.get("drawer", "")).strip().upper()[-1:]
                if r == self.room and d == self.drawer:
                    self.opened = True
                    return f"Inside drawer {d} you find a card: code word '{self.code}'.", False, False
                return f"Drawer {d} in room {r} is empty.", False, False
            if name == "submit":
                ok = str(args.get("code", "")).strip().strip("'\"").lower() == self.code
                return ("Correct!" if ok else "Wrong code."), True, ok
        except Exception as e:  # malformed args
            return f"Error: {e}", False, False
        return f"Unknown tool {name}.", False, False


ALFWORLD_SPLITS = {"train": "data_path", "eval_in_distribution": "eval_id_data_path",
                   "eval_out_of_distribution": "eval_ood_data_path"}
# The keys AlfredTWEnv reads, from alfworld's configs/base_config.yaml (that file is not shipped in
# the pip package). $ALFWORLD_DATA defaults to ~/.cache/alfworld (alfworld sets it on import).
ALFWORLD_DEFAULT_CONFIG = {
    "dataset": {"data_path": "$ALFWORLD_DATA/json_2.1.1/train",
                "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
                "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
                "num_train_games": -1, "num_eval_games": -1},
    "env": {"type": "AlfredTWEnv", "task_types": [1, 2, 3, 4, 5, 6], "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False, "expert_type": "handcoded"},
}


def _alfworld_game_files(config_path: str | None, split: str) -> list[str]:
    if split not in ALFWORLD_SPLITS:
        raise ValueError(f"unknown ALFWorld split {split!r}; expected one of {sorted(ALFWORLD_SPLITS)}")
    if config_path:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = copy.deepcopy(ALFWORLD_DEFAULT_CONFIG)
    # import the class from its module: alfworld>=0.4 no longer re-exports it from the package
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    config["dataset"]["num_train_games"] = config["dataset"]["num_eval_games"] = -1  # we sort, then index
    data_dir = os.path.expandvars(config["dataset"][ALFWORLD_SPLITS[split]])
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"ALFWorld data not found at {data_dir!r}: run `alfworld-download` "
                                "(data goes to $ALFWORLD_DATA, default ~/.cache/alfworld) or pass --alfworld-config")
    return AlfredTWEnv(config, train_eval=split).game_files


def _alfworld_tw_env(max_steps: int):
    """One TextWorld wrapper stack (as textworld.gym builds it) that can load any game file."""
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from textworld.envs.wrappers import Filter, GenericEnvironment, Limit
    env = GenericEnvironment(textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"]))
    env = Limit(env, max_episode_steps=max_steps)
    return Filter(AlfredInfos(AlfredDemangler(shuffle=False)(env)))


class ALFWorldEnv(Env):
    """ALFWorld text environment (requires `pip install alfworld` and `alfworld-download`).

    Task i is always game i of the split's game files in sorted order, so repeated runs, resume and
    per-task budgets refer to the same game (ALFWorld's own env hands out a shuffled game per reset).
    Single tool `act(command)`; observations include the admissible commands, as in common ReAct
    setups. Success = info['won'].
    """

    system = ("You are an agent in a household. Solve the task by issuing one text command per turn "
              "with the `act` tool. Use only admissible commands.")
    tools = [tool("act", "Issue a text command in the environment.", command="an admissible command")]
    max_rounds = 50

    def __init__(self, config_path: str | None = None, split: str = "eval_out_of_distribution",
                 *, game_files: list[str] | None = None, tw_env=None):
        if game_files is None:
            game_files = _alfworld_game_files(config_path, split)
        elif split not in ALFWORLD_SPLITS:
            raise ValueError(f"unknown ALFWorld split {split!r}; expected one of {sorted(ALFWORLD_SPLITS)}")
        self.game_files = sorted(game_files)
        if not self.game_files:   # ALFWorld itself would spin forever on an empty game list
            raise RuntimeError("no ALFWorld games found")
        self.tw = tw_env if tw_env is not None else _alfworld_tw_env(self.max_rounds)
        self.game_id = None

    def __len__(self):
        return len(self.game_files)

    def reset(self, i: int) -> str:
        game = self.game_files[i]
        self.tw.load(game)
        obs, info = self.tw.reset()
        self.game_id = "/".join(game.split(os.sep)[-3:-1])     # <task dir>/<trial dir>
        self.admissible = list(info["admissible_commands"])
        return obs + "\n\nAdmissible commands: " + ", ".join(self.admissible)

    def step(self, name, args):
        if name != "act":
            return f"Unknown tool {name}. Use act(command).", False, False
        cmd = str(args.get("command", "")).strip()
        obs, _, done, info = self.tw.step(cmd)
        self.admissible = list(info["admissible_commands"])
        won = bool(info["won"])
        text = obs + "\n\nAdmissible commands: " + ", ".join(self.admissible)
        return text, bool(done) or won, won


def make_env(name: str, **kw) -> Env:
    if name == "vault":
        return VaultEnv(**kw)
    if name == "alfworld":
        return ALFWorldEnv(**kw)
    raise ValueError(name)
