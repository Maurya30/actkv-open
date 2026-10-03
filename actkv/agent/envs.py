"""Agent environments with a common interface.

    env.system, env.tools          prompt + tool schemas
    env.reset(i) -> task text       start task i
    env.step(name, args) -> (observation, done, success)
"""
from __future__ import annotations

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


class ALFWorldEnv(Env):
    """ALFWorld text environment (requires `pip install alfworld` and `alfworld-download`).

    Single tool `act(command)`; observations include the admissible commands, as in common
    ReAct setups. Success = info['won'].
    """

    system = ("You are an agent in a household. Solve the task by issuing one text command per turn "
              "with the `act` tool. Use only admissible commands.")
    tools = [tool("act", "Issue a text command in the environment.", command="an admissible command")]
    max_rounds = 50

    def __init__(self, config_path: str, split: str = "eval_out_of_distribution", n_tasks: int = 134):
        import alfworld.agents.environment as environment  # lazy
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        env_cls = getattr(environment, config["env"]["type"])
        self.env = env_cls(config, train_eval=split).init_env(batch_size=1)
        self.n_tasks = n_tasks

    def __len__(self):
        return self.n_tasks

    def reset(self, i: int) -> str:
        # ALFWorld iterates its own game list; `i` is only used for bookkeeping
        obs, info = self.env.reset()
        self.admissible = info["admissible_commands"][0]
        text = obs[0]
        return text + "\n\nAdmissible commands: " + ", ".join(self.admissible)

    def step(self, name, args):
        if name != "act":
            return f"Unknown tool {name}. Use act(command).", False, False
        cmd = str(args.get("command", "")).strip()
        obs, _, dones, info = self.env.step([cmd])
        self.admissible = info["admissible_commands"][0]
        won = bool(info["won"][0])
        text = obs[0] + "\n\nAdmissible commands: " + ", ".join(self.admissible)
        return text, bool(dones[0]) or won, won


def make_env(name: str, **kw) -> Env:
    if name == "vault":
        return VaultEnv(**kw)
    if name == "alfworld":
        return ALFWorldEnv(**kw)
    raise ValueError(name)
