import re
import sys

# warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

import numpy as np
import hydra
from omegaconf import OmegaConf
import dreamerv3
import embodied
from envs.maniskill import make_maniskill_env
from envs.exceptions import UnknownTaskError
from functools import partial as bind


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


# class WandBOutput:
#     def __init__(self, cfg):
#         # self._pattern = re.compile(pattern)
#         wandb.init(
#             project=cfg.wandb_project,
#             entity=cfg.wandb_entity,
#             name=str(cfg.seed),
#             group=self.cfg_to_group(cfg),
#             tags=self.cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
#             config=OmegaConf.to_container(cfg, resolve=True),
#         )
#         self._wandb = wandb

#     def cfg_to_group(self, cfg, return_list=False):
#         """Return a wandb-safe group name for logging. Optionally returns group name as list."""
#         lst = [cfg.task, cfg.obs_mode, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
#         return lst if return_list else "-".join(lst)

#     def __call__(self, summaries):
#         bystep = collections.defaultdict(dict)
#         for step, name, value in summaries:
#             try:
#                 bystep[step][name] = float(value)
#             except:
#                 continue
#         for step, metrics in bystep.items():
#             self._wandb.log(metrics, step=step)


def rand_str(length=6):
    chars = "abcdefghijklmnopqrstuvwxyz"
    return "".join(np.random.choice(list(chars)) for _ in range(length))


def train(cfg):
    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(
        {
            **dreamerv3.Agent.configs["size25m"],
            "task": cfg.task,
            "logdir": f"{cfg.logging_dir}/{cfg.task}-{cfg.exp_name}-{cfg.seed}-{rand_str()}",
            "seed": cfg.seed,
            # "run.train_ratio": 512,
            # "run.log_every": 120,  # Seconds
            "run.steps": cfg.steps,
            "run.eval_every": cfg.eval_freq,
            "run.eval_eps": cfg.eval_episodes,
            "run.num_envs": 1,
            # "batch_size": 16,
            # "jax.prealloc": False,
            "replay.size": 1_000_000,
        }
    )
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    def make_agent(config):
        env = make_env(config)
        agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
        env.close()
        return agent

    def make_env(cfg, env_id=0):
        """
        Make environment.
        """
        from embodied.envs import from_gym

        env = None
        try:
            env = make_maniskill_env(cfg)
        except UnknownTaskError:
            pass
        if env is None:
            raise UnknownTaskError(cfg.task)
        env = from_gym.FromGym(env, obs_key="rgb")
        env = dreamerv3.wrap_env(env, cfg)
        return env

    def make_logger(model_config, project_config):
        logdir = embodied.Path(model_config.logdir)
        loggers = [
            embodied.logger.TerminalOutput(model_config.filter),
            embodied.logger.TensorBoardOutput(logdir),
        ]
        if project_config.wandb_enable:
            loggers.append(
                embodied.logger.WandBOutput(
                    str(project_config.seed),
                    ".*",
                    project=project_config.wandb_project,
                    entity=project_config.wandb_entity,
                    group=cfg_to_group(project_config),
                    tags=cfg_to_group(project_config, return_list=True)
                    + [f"seed:{project_config.seed}"],
                    config=OmegaConf.to_container(project_config, resolve=True),
                )
            )

        return embodied.Logger(
            embodied.Counter(),
            loggers,
        )

    def make_replay(config):
        return embodied.replay.Replay(
            length=config.batch_length,
            capacity=config.replay.size,
            directory=embodied.Path(config.logdir) / "replay",
            online=config.replay.online,
        )

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_context=config.replay_context,
    )

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config),
        bind(make_env, config),
        bind(make_logger, config, cfg),
        args,
    )


@hydra.main(version_base=None, config_name="config", config_path=".")
def launch(cfg: dict):
    sys.argv = sys.argv[:1]
    try:
        train(cfg)
    # account for free() invalid pointer error
    except Exception as e:
        print("Error in train.py:")
        print(e)
        pass


if __name__ == "__main__":
    launch()
