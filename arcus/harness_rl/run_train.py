from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

try:
    import gym as old_gym
except Exception:
    old_gym = None

class TrainingProgressCallback(BaseCallback):

    BAR_WIDTH = 30

    def __init__(
        self,
        total_timesteps: int,
        algo: str,
        env_id: str,
        seed: int,
        update_interval: int = 2000,
    ):
        super().__init__(verbose=0)
        self.total_timesteps   = total_timesteps
        self.algo              = algo.upper()
        self.env_id            = env_id
        self.seed              = seed
        self.update_interval   = update_interval
        self._last_update_step = 0
        self._start_time: Optional[float] = None
        self._last_ep_rewards: list[float] = []

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._render()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._last_ep_rewards.append(float(info["episode"]["r"]))
                if len(self._last_ep_rewards) > 50:
                    self._last_ep_rewards.pop(0)

        if self.num_timesteps - self._last_update_step >= self.update_interval:
            self._render()
            self._last_update_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        n   = self.num_timesteps
        tot = self.total_timesteps
        pct = min(n / max(tot, 1), 1.0)

        filled = int(self.BAR_WIDTH * pct)
        bar    = "█" * filled + "░" * (self.BAR_WIDTH - filled)

        elapsed = time.time() - (self._start_time or time.time())
        if pct > 0.001 and elapsed > 0:
            eta_sec = elapsed / pct * (1.0 - pct)
            eta_str = _fmt_duration(eta_sec)
        else:
            eta_str = "?"

        if self._last_ep_rewards:
            rew_str = f"  rew={np.mean(self._last_ep_rewards[-20:]):+.1f}"
        else:
            rew_str = ""

        label = f"[{self.algo} | {self.env_id} | seed={self.seed}]"
        line  = (
            f"\r{label}  {pct*100:5.1f}%  [{bar}]  "
            f"{n//1000}k/{tot//1000}k steps  ETA {eta_str}{rew_str}  "
        )
        sys.stderr.write(line)
        sys.stderr.flush()


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m:02d}m"

def _ensure_atari_registered():
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass

def _is_image_obs(space: gym.spaces.Space) -> bool:
    if not isinstance(space, gym.spaces.Box):
        return False
    if space.dtype is None:
        return False
    if len(space.shape) != 3:
        return False
    return True


def _auto_policy_for_env(env) -> str:
    obs_space = env.observation_space
    if _is_image_obs(obs_space):
        return "CnnPolicy"
    if isinstance(obs_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"

def _make_procgen_vec_env(env_id: str, seed: int, n_envs: int):
    if old_gym is None:
        raise RuntimeError("Old gym is required for procgen but not installed/importable.")
    import procgen

    from arcus.harness_rl.run_eval import GymOldToGymnasiumEnv

    def _thunk():
        raw = old_gym.make(env_id)
        while hasattr(raw, "env") and type(raw).__name__ in (
            "OrderEnforcing", "PassiveEnvChecker", "EnvChecker"
        ):
            raw = raw.env
        return GymOldToGymnasiumEnv(raw)

    return DummyVecEnv([_thunk for _ in range(max(1, int(n_envs)))])


def _make_env(env_id: str, seed: int, n_envs: int):
    """Central env factory."""
    if env_id.startswith("procgen:"):
        return _make_procgen_vec_env(env_id, seed=seed, n_envs=n_envs)

    if env_id.startswith("ALE/"):
        _ensure_atari_registered()
        venv = make_atari_env(env_id, n_envs=max(1, int(n_envs)), seed=seed)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)
        return venv
    return make_vec_env(env_id, n_envs=max(1, int(n_envs)), seed=seed)

def _build_model(
    algo: str,
    env,
    *,
    device: str,
    tb_log: Optional[str],
    verbose: int,
    policy: str,
):
    algo = algo.lower()

    common = dict(
        policy=policy,
        env=env,
        device=device,
        verbose=verbose,
        tensorboard_log=tb_log,
    )

    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO(**common)

    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C(**common)

    if algo == "dqn":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(f"DQN requires Discrete action space, got {env.action_space}")
        from stable_baselines3 import DQN
        return DQN(**common)

    if algo == "qrdqn":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(f"QR-DQN requires Discrete action space, got {env.action_space}")
        from sb3_contrib import QRDQN
        return QRDQN(**common)

    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC(**common)

    if algo == "td3":
        from stable_baselines3 import TD3
        return TD3(**common)

    if algo == "ddpg":
        from stable_baselines3 import DDPG
        return DDPG(**common)

    if algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO(**common)

    raise ValueError(
        f"Unsupported algo '{algo}'. "
        f"Supported: ppo, a2c, dqn, qrdqn, sac, td3, ddpg, trpo"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",         required=True)
    ap.add_argument("--algo",        required=True)
    ap.add_argument("--timesteps",   type=int, default=300_000)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--n_envs",      type=int, default=1)
    ap.add_argument("--policy",      default="auto")
    ap.add_argument("--verbose",     default="0")
    ap.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the progress bar (useful when stdout is piped or in CI)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.env, seed=int(args.seed), n_envs=max(1, int(args.n_envs)))

    policy = args.policy
    if policy == "auto":
        policy = _auto_policy_for_env(env)

    tb_log = str(out_dir / "tb")

    model = _build_model(
        args.algo,
        env,
        device=str(args.device),
        tb_log=tb_log,
        verbose=int(args.verbose),
        policy=policy,
    )

    callbacks: list[BaseCallback] = []
    if not args.no_progress:
        callbacks.append(
            TrainingProgressCallback(
                total_timesteps=int(args.timesteps),
                algo=args.algo,
                env_id=args.env,
                seed=int(args.seed),
            )
        )

    model.learn(
        total_timesteps=int(args.timesteps),
        callback=callbacks if callbacks else None,
    )

    env_safe = args.env.replace("/", "").replace(":", "_")
    zip_path = out_dir / f"{args.algo.lower()}_{env_safe}.zip"
    model.save(zip_path)
    env.close()

    print(
        f"[OK] trained : env={args.env}  algo={args.algo}  "
        f"seed={args.seed}  timesteps={args.timesteps}  policy={policy}"
    )
    print(f"[OK] model   : {zip_path}")
    print(f"[OK] tb logs : {tb_log}")


if __name__ == "__main__":
    main()
