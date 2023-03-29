from random import random
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import gym
from gym.wrappers import Monitor, RecordVideo
import numpy as np
import pprint, ray, os
# from ipdb import set_trace as st
from hyperopt import hp
from ray.air import CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.skopt import SkOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.async_hyperband import ASHAScheduler

from ray import tune, air
from ray.tune import grid_search
from ray.tune.logger import pretty_print

# Start a new instance of Ray (when running this tutorial locally) or
# connect to an already running one (when running this tutorial through Anyscale).

# In case you encounter the following error during our tutorial: `RuntimeError: Maybe you called ray.init twice by accident?`
# Try: `ray.shutdown() + ray.init()` or `ray.init(ignore_reinit_error=True)

# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

# Specify a very simple config, defining our environment and some environment
# options (see environment.py).
from ray.tune.registry import register_env

import pandas as pd
import matplotlib.pyplot as plt

# from affectameute_env_blocage_reservoir_pygame_v2 import AffectaMeuteEnv
# register_env("affectameuteEnvBlocageReservoirv2", lambda config: AffectaMeuteEnv())

# from affectameute_env_deplacement_aleatoire_dist_v2 import AffectaMeuteEnv
# register_env("affectameuteEnvDeplacementAleatoireDistv2", lambda config: AffectaMeuteEnv())

# from affectameute_env_placement_aleatoire_reward_0_v2 import AffectaMeuteEnv
# register_env("affectameuteEnvv2", lambda config: AffectaMeuteEnv())

# from affectameute_env_aleatoire_sanction_paroi_v2 import AffectaMeuteEnv
# register_env("affectameuteEnvDeplacementAleatoirev2", lambda config: AffectaMeuteEnv())

from affectameute_v1 import AffectaMeuteEnv
register_env("rewardContinuX", lambda config: AffectaMeuteEnv())

gym.envs.register(
    id='affecta-v0',
    entry_point='affectameute_v1:AffectaMeuteEnv',
    # kwargs={'n_agents': 2, 'full_observable': False, 'step_cost': -0.2}
    # It has a step cost of -0.2 now
)

ray.init(ignore_reinit_error=True, log_to_driver=False)

def compute_actions(config, checkpoint):
    env = gym.make('affecta-v0')
    # Créer un objet Monitor pour enregistrer la vidéo
    # video_dir = '/home/cytech/Videos/videos_from_wrappers'
    # env = Monitor(env, video_dir, force=True)

    rllib_trainer = PPOTrainer(config=config)
    rllib_trainer.restore(checkpoint)
    # On entraine sur quelques épisodes

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = rllib_trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()

    env.close()
    print("Environnement fermé")
    print("reward : ",episode_reward)
    # env = env.make('affecta-v0')
    # env = Monitor(env, video_dir, force=True)

def unit_ppo(config, nb_episode, checkpoint):
    rllib_trainer = PPOTrainer(config)
    rllib_trainer.restore(checkpoint)

    # On entraine sur quelques épisodes
    N = nb_episode
    results = []
    episode_data = []

    for n in range(N):
        result = rllib_trainer.train()
        results.append(result)

        episode = {
            "n": n,
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
        }

        episode_data.append(episode)

        print(f'\nIteration {n} : ')
        print(f'Min reward: {episode["episode_reward_min"]}')
        print(f'Max reward: {episode["episode_reward_max"]}')
        print(f'Mean reward: {episode["episode_reward_mean"]}')

    # Affichage des perfs
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(data=episode_data)
    df
    df.plot(x="n", y=["episode_reward_mean", "episode_reward_min", "episode_reward_max"], secondary_y=False).legend(
        bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

# fonction qui permettra de récupérer le dico best_conf si besoin
def save(conf):
    fichier = open("conf.txt", "w")  # Créer le fichier s'il n'existe pas
    fichier.write(str(conf))  # Écrit la valeur de la variable conf dans le fichier
    fichier.close()

def explore(config):
    # on doit avoir assez de timesteps pour faire un sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # nécesite au moins un sgd
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


def hyperopt():
    # Configuration pour la recherche d'hyperparamètres via un algorithme hyper_optimization_search

    config = {
                 "env": "affecta",
                 "sgd_minibatch_size": 1000,
                 "num_sgd_iter": 1000,
                 "lr": tune.uniform(5e-6, 5e-2),
                 "lambda": tune.uniform(0.6, 0.99),
                 "vf_loss_coeff": tune.uniform(0.6, 0.99),
                 "kl_target": tune.uniform(0.001, 0.01),
                 "kl_coeff": tune.uniform(0.5, 0.99),
                 "entropy_coeff": tune.uniform(0.001, 0.01),
                 "clip_param": tune.uniform(0.4, 0.99),
                 "train_batch_size": 2, # taille de l'épisode
                 # "monitor": True,
                 # "model": {"free_log_std": True},
                 "num_workers": 4,
                 "num_gpus": 0,
                 # "rollout_fragment_length":3
                 # "batch_mode": "complete_episodes"
             }

    current_best_params = [{
        'lr': 5e-4,
    }]

    config = explore(config)
    optimizer = HyperOptSearch(metric="episode_reward_mean", mode="max", n_initial_points=1, random_state_seed=7, space=config)

    # optimizer = ConcurrencyLimiter(optimizer, max_concurrent=4)

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",  # the metric we want to study
            # mode="max",  # maximize the metric
            search_alg=optimizer,
            # num_samples will repeat the entire config 'num_samples' times == Number of trials dans l'output 'Status'
            num_samples=100,
        ),
        run_config=air.RunConfig(stop={"training_iteration": 20}, local_dir="lunar_tune"),
        # limite le nombre d'épisode pour chaque croisement d'hyperparamètres

    )
    results = tuner.fit()

    # analysis.get_best_trial(metric="episode_reward_mean", mode="max")
    # accuracy = 1. - analysis.best_result["eval-error"]
    # print("\n\n\nBest congig :", analysis.get_best_config(metric="episode_reward_mean", mode="max"))
    # print(f"\nAccuracy : {accuracy:.4f}")
    # print("\nstats : ", analysis.stats())


    # affichage de la meilleure configuration
    best_conf = results.get_best_result(metric="episode_reward_mean", mode="max").config
    print(f"\n ##############################################\n Meilleure configuration : {best_conf}\n ##############################################\n")
    save(best_conf)

    # évaluation du modèle
    checkpoint = results.get_dataframe(metric="episode_reward_mean", mode="max").checkpoint
    print(f"\n ##############################################\n Checkpoint : {checkpoint}\n ##############################################\n")

    # compute_actions(best_conf, checkpoint)
    unit_ppo(best_conf, 30, checkpoint)


    # plot des performances
    # result_df = results.get_best_result().metrics_dataframe
    # results.get_best_result().metrics_dataframe.plot("training_iteration", "episode_reward_mean")


def unit_ppo_default(config, nb_episode, checkpoint = None):


    rllib_trainer = PPOTrainer(config)
    if checkpoint != None:
        rllib_trainer.restore(checkpoint)

    # On entraine sur quelques épisodes
    N = nb_episode
    results = []
    episode_data = []

    for n in range(N):
        result = rllib_trainer.train()
        results.append(result)

        episode = {
            "n": n,
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
        }

        episode_data.append(episode)

        print(f'\nIteration {n} : ')
        print(f'Min reward: {episode["episode_reward_min"]}')
        print(f'Max reward: {episode["episode_reward_max"]}')
        print(f'Mean reward: {episode["episode_reward_mean"]}')

        if n % 200 == 0 and n != 0:
            rllib_trainer.save(f"v3{n}")
            # # Affichage des perfs
            # df = pd.DataFrame(data=episode_data)
            # df
            # df.plot(x="n", y=["episode_reward_mean", "episode_reward_min", "episode_reward_max"], secondary_y=False).legend(
            #     bbox_to_anchor=(1.05, 1.0), loc='upper left')
            # plt.show()

def unit_hyperopt():
    # Configuration pour la recherche d'hyperparamètres via un algorithme hyper_optimization_search

    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = 3
    config["cpu_per_worker"] = 1
    config["num_sgd_iter"] = 100
    config["sgd_minibatch_size"] = 1000
    config["model"]["fcnet_hiddens"] = [20, 20, 20]
    config["train_batch_size"] = 2000
    config["framework"] = "torch"
    config["env"] = "affectameuteEnvDeplacementAleatoireParoiv2"

    config = explore(config)
    # optimizer = HyperOptSearch(metric="episode_reward_mean", mode="max", n_initial_points=1, random_state_seed=7,
    #                            space=config)

    # optimizer = ConcurrencyLimiter(optimizer, max_concurrent=4)
    #
    tuner = tune.run(
        "PPO",
        tune_config=tune.TuneConfig(
            # metric="episode_reward_mean",  # the metric we want to study
            # mode="max",  # maximize the metric
            # search_alg=optimizer,
            # num_samples will repeat the entire config 'num_samples' times == Number of trials dans l'output 'Status'
            num_samples=1,
            # max_concurrent_trials=2,

        ),
        param_space=config,
        run_config=air.RunConfig(stop={"training_iteration": 100000000},
                                 local_dir="train_aleatoire_sanction_paroi",
                                 checkpoint_config=air.CheckpointConfig(checkpoint_frequency=200),
                                 failure_config=air.FailureConfig(max_failures=5))
    )

    # limite le nombre d'épisode pour chaque croisement d'hyperparamètres
    results = tuner.fit()

def train(config, reporter):
    # Initialise le SummaryWriter pour enregistrer les métriques dans Tensorboard
    writer = SummaryWriter()

    # Boucle d'entrainement
    for i in range(config["num_iterations"]):
        # Entrainement du modèle
        train_loss = ...
        train_acc = ...

        # Evaluation du modèle
        val_loss = ...
        val_acc = ...

        # Mise a jour des métriques
        reporter(mean_accuracy=train_acc, mean_loss=train_loss, val_accuracy=val_acc, val_loss=val_loss)

        # Enregistrement des métriques dans Tensorboard
        writer.add_scalar("mean_accuracy", train_acc, i)
        writer.add_scalar("mean_loss", train_loss, i)
        writer.add_scalar("val_accuracy", val_acc, i)
        writer.add_scalar("val_loss", val_loss, i)

        # Affichage des métriques dans la console
        if i % 100 == 0:
            print(
                f"Iteration {i}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Fermeture du SummaryWriter
    writer.close()

def ray_run():
    # Configuration pour la recherche d'hyperparamètres via un algorithme hyper_optimization_search

    config = DEFAULT_CONFIG.copy()
    config["num_workers"] = 6
    config["cpu_per_worker"] = 1
    config["num_sgd_iter"] = 100
    config["sgd_minibatch_size"] = 1000
    # config["model"]["fcnet_hiddens"] = [14, {"lstm": 64}]
    config["model"]["fcnet_hiddens"] = [5]
    config["model"]["use_lstm"] = True
    config["train_batch_size"] = 2000
    config["framework"] = "torch"
    config["env"] = "rewardContinuX"

    config = explore(config)

    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 800},
        local_dir="run_reward_continu_X",
        checkpoint_freq=200,
        max_failures=5,
        num_samples=1,
        # restore='run_reward_continu_5/PPO/PPO_rewardContinu_70eda_00004_4_fcnet_hiddens=40_30_20_10_20_30_40_2023-03-03_14-22-06/checkpoint_000200'
        )

def compute_several_actions(config, checkpoint, n):
    for i in range(n):
        compute_actions(config, checkpoint)


config = DEFAULT_CONFIG.copy()
config["num_workers"] = 6
config["cpu_per_worker"] = 1
config["num_sgd_iter"] = 100
config["sgd_minibatch_size"] = 1000
# config["model"]["fcnet_hiddens"] = [14, {"lstm": 64}]
config["model"]["fcnet_hiddens"] = [6, 6]
# config["model"]["use_lstm"] = True
config["train_batch_size"] = 2000
config["framework"] = "torch"
config["env"] = "rewardContinuX"

checkpoint_path = 'run_reward_continu_X/PPO/PPO_rewardContinuX_e328a_00000_0_2023-03-16_14-46-00/checkpoint_000200'
ray_run()
# compute_several_actions(config,checkpoint_path, 5)

# unit_hyperopt()
# unit_ppo_default(config, 10000000000000000000)