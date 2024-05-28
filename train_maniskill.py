import re
import sys
import numpy as np
import hydra
from omegaconf import OmegaConf
import dreamerv3
import embodied
from envs.maniskill import make_maniskill_env
from envs.exceptions import UnknownTaskError
from functools import partial as bind
from embodied import wrappers
from omegaconf import DictConfig


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    task_name = tasks_to_name(cfg.tasks)
    lst = [task_name, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


def tasks_to_name(tasks):
    if len(tasks) == 1:
        return tasks[0]
    return "multitask"


def rand_str(length=6):
    chars = "abcdefghijklmnopqrstuvwxyz"
    return "".join(np.random.choice(list(chars)) for _ in range(length))


def train(cfg):
    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.Agent.configs["defaults"])
    config = config.update(
        {
            **dreamerv3.Agent.configs["size25m"],
            **dreamerv3.Agent.configs["maniskill"],
            "logdir": f"{cfg.logging_dir}/{tasks_to_name(cfg.tasks)}-{cfg.exp_name}-{cfg.seed}-{rand_str()}",
            "seed": cfg.seed,
            "run.steps": cfg.steps,
            "run.eval_every": cfg.eval_freq,
            "run.eval_eps": cfg.eval_episodes,
            "replay.size": 1_000_000,
        }
    )
    config = embodied.Flags(config).parse()
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / "config.yaml")

    def make_agent(config, hydra_cfg):
        env = make_env(config, hydra_cfg)
        agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
        env.close()
        return agent

    def make_env(cfg, hydra_cfg, env_id=0):
        """
        Make environment.
        """
        from embodied.envs import from_gym

        env = None
        try:
            env = make_maniskill_env(cfg, hydra_cfg)
        except UnknownTaskError:
            pass
        if env is None:
            raise UnknownTaskError(tasks_to_name(cfg.tasks))
        env = from_gym.FromGym(env)
        for name, space in env.act_space.items():
            if name == "reset":
                continue
            elif not space.discrete:
                env = wrappers.NormalizeAction(env, name)
                if cfg.wrapper.discretize:
                    env = wrappers.DiscretizeAction(env, name, cfg.wrapper.discretize)
        if cfg.wrapper.checks:
            env = wrappers.CheckSpaces(env)
        for name, space in env.act_space.items():
            if not space.discrete:
                env = wrappers.ClipAction(env, name)
        return env

    def make_logger(model_config, project_config):
        loggers = [
            embodied.logger.TerminalOutput(".*"),
        ]
        if project_config.wandb_enable:
            loggers.append(
                embodied.logger.WandBOutput(
                    str(project_config.seed),
                    model_config.filter,
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

    embodied.run.train_overwrite(
        bind(make_agent, config, cfg),
        bind(make_replay, config),
        bind(make_env, config, cfg),
        bind(make_logger, config, cfg),
        args,
    )


@hydra.main(version_base=None, config_name="config", config_path=".")
def main(cfg: DictConfig):
    sys.argv = sys.argv[:1]
    train(cfg)



if __name__ == "__main__":
    main()
