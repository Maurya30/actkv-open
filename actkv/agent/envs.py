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
    """Offline memory task with long, noisy observations (useful for smoke tests and quick ablations).

    One room holds a note with the vault code; the note crumbles after it is read once. The note
    also says where a trail starts. Trail rooms show nothing special until the note has been read,
    and only in order: each points to the next, and the last one holds the vault. Submitting works
    only at the vault and only once. So the code has to be carried across at least `trail` + 1
    rounds of long observations, and with len(WORDS) * 10,000 possible codes a blind guess
    essentially never works.
    """

    system = ("You are an agent exploring a building. Use the tools to find the vault code, follow the "
              "trail to the vault and open it. Call exactly one tool per turn.")
    tools = [
        tool("search_room", "Look around a room.", room="room name, e.g. 'room 3'"),
        tool("submit", "Enter the code on the vault keypad (one attempt).", code="the code, e.g. 'amber-0420'"),
    ]
    WORDS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo",
             "juniper", "krypton", "lagoon", "marble", "nebula", "obsidian", "prairie", "quartz"]
    ITEMS = ["chair", "lamp", "box", "rug", "book", "vase", "clock", "plant", "mug", "poster",
             "cable", "jar", "tray", "fan", "basket", "mirror", "kettle", "towel"]

    def __init__(self, n_tasks: int = 50, n_rooms: int = 8, trail: int = 4, filler: int = 40, seed: int = 0):
        if trail < 1 or trail + 1 > n_rooms:
            raise ValueError("need 1 <= trail <= n_rooms - 1")
        self.n_tasks, self.n_rooms, self.trail, self.filler, self.seed = n_tasks, n_rooms, trail, filler, seed
        self.max_rounds = n_rooms + trail + 4

    def __len__(self):
        return self.n_tasks

    def reset(self, i: int) -> str:
        rng = random.Random(self.seed * 100_003 + i)
        rooms = rng.sample(range(1, self.n_rooms + 1), self.trail + 1)
        self.note_room, self.trail_rooms = rooms[0], rooms[1:]
        self.code = f"{rng.choice(self.WORDS)}-{rng.randrange(10_000):04d}"
        self.salt = rng.randrange(10**9)
        self.note_read, self.reached, self.at_vault = False, 0, False
        return (f"The building has rooms 1 to {self.n_rooms}. A note with the vault code is in one of the "
                "rooms. Find and read it, follow the trail it describes to the vault, and submit the code there.")

    def _search(self, r: int) -> str:
        extra = []
        if r == self.note_room:
            if not self.note_read:
                self.note_read = True
                extra.append(f"a note: 'The vault code is {self.code}. The trail to the vault starts in room "
                             f"{self.trail_rooms[0]}.' The note crumbles to dust as you finish reading it")
            else:
                extra.append("a small pile of dust")
        if self.note_read and r in self.trail_rooms:
            k = self.trail_rooms.index(r)
            self.reached += k == self.reached         # trail rooms count only in order
            if k < self.reached:
                extra.append(f"an arrow pointing to room {self.trail_rooms[k + 1]}" if k + 1 < self.trail
                             else "the vault, with a keypad")
        self.at_vault = self.reached == self.trail and r == self.trail_rooms[-1]
        rng = random.Random(self.salt * 131 + r)
        items = [f"{rng.choice(self.ITEMS)} #{rng.randrange(1000)}" for _ in range(self.filler)]
        for e in extra:
            items.insert(rng.randrange(len(items) + 1), e)
        return f"Room {r} contains: " + "; ".join(items) + "."

    def step(self, name, args):
        try:
            if name == "search_room":
                r = int("".join(ch for ch in str(args.get("room", "")) if ch.isdigit()) or -1)
                if not 1 <= r <= self.n_rooms:
                    return f"There is no {args.get('room')}.", False, False
                return self._search(r), False, False
            if name == "submit":
                if not self.at_vault:
                    return "There is no vault here.", False, False
                ok = str(args.get("code", "")).strip().strip("'\"").lower() == self.code
                return ("The vault opens." if ok else "Wrong code. The keypad locks."), True, ok
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
