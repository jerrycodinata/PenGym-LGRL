import time
from os.path import isfile
import logging
import sys
import getopt
import numpy as np
import random
import gymnasium as gym
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from sb3_contrib import MaskablePPO
import pengym.utilities as utils
from pengym.storyboard import Storyboard
from lgrl.subgoal_managers import OracleSubgoalManager
from lgrl.common_utilities import make_env, make_custom_env
from lgrl.action_mask import custom_mask_fn
from lgrl.evaluation import ConvergenceCallback

storyboard = Storyboard()

# Agent types
AGENT_TYPE_PPO = "ppo"
AGENT_TYPE_LGRL = "lgrl"
DEFAULT_AGENT_TYPE = AGENT_TYPE_PPO

# Other constants
MAX_STEPS = 150 # Max number of pentesting steps
TOTAL_EPISODES = 100
EVAL_EPISODES = 100
TOTAL_TIMESTEPS = MAX_STEPS * TOTAL_EPISODES
FRAME_MEMORY = 4 # Number of frames to stack for PPO agent
WINDOW_SIZE = 5 # Number of episodes to consider for convergence calculation
MARGIN = 2 # Margin of steps above ideal steps to consider as convergence
RENDER_OBS_STATE = False

IDEAL_STEPS = {
    "tiny": 6,
    "tiny-small": 7,
    "tiny-hard": 5,
    "small-linear": 12,
    "small-honeypot": 8,
    "medium": 8,
    "medium-single-site": 4,
    "medium-multi-site": 7,
}

def normalize_firewall_collections(env):
    """Normalize firewall service collections to lists for compatibility.

    Generated NASim scenarios may provide firewall values as sets, while some
    PenGym code assumes list concatenation.
    """
    try:
        base_env = env.unwrapped
        network = getattr(base_env, "network", None)
        firewall = getattr(network, "firewall", None)

        if isinstance(firewall, dict):
            for link, services in firewall.items():
                if not isinstance(services, list):
                    if isinstance(services, (set, tuple)):
                        firewall[link] = list(services)
                    else:
                        try:
                            firewall[link] = list(services)
                        except TypeError:
                            firewall[link] = [services]
    except Exception:
        # Best-effort normalization only; preserve env behavior if not applicable.
        pass

    return env

def make_eval_env(scenario_name, seed, **kwargs):
    return lambda: normalize_firewall_collections(make_env(scenario_name, seed=seed, **kwargs))

def make_env_with_seed(scenario_name, train_seeds, **kwargs):
    def _init():
        seed = random.choice(train_seeds)
        return normalize_firewall_collections(make_env(scenario_name, seed=seed, **kwargs))
    return _init

def make_env_reseedable(scenario_name, seed_pool, **kwargs):
    class _Env(gym.Env):
        def __init__(self):
            self.scenario_name = scenario_name
            self.seed_pool = seed_pool
            self.kwargs = kwargs

            self.env = make_env(
                scenario_name,
                seed=random.choice(seed_pool),
                **kwargs
            )
            self.env = normalize_firewall_collections(self.env)

            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        def reset(self, **kwargs):
            seed = random.choice(self.seed_pool)

            self.env = make_env(
                self.scenario_name,
                seed=seed,
                **self.kwargs
            )
            self.env = normalize_firewall_collections(self.env)
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

        def render(self, *args, **kwargs):
            return self.env.render(*args, **kwargs)

        def action_masks(self):
            # Expose masking method at top-level so DummyVecEnv.has_attr/env_method can find it.
            return self.env.get_wrapper_attr("action_masks")()
        
        def __getattr__(self, name):
            # Delegate through wrapper-aware lookup to preserve attributes hidden by Monitor/TimeLimit.
            return self.env.get_wrapper_attr(name)

    return _Env

def run_agent(agent_type, scenario_path=None, scenario_name=None, model_path=None):
    scenario_name = scenario_name if scenario_name is not None else scenario_path.split("/")[-1].split(".")[0]
    subgoal_manager = None
    env_fn = None
    model = None
    convergence_speed = -1

    train_seeds = list(range(100))

    if scenario_path is not None:
        if agent_type == AGENT_TYPE_LGRL:
            subgoal_manager = OracleSubgoalManager(utils=utils, storyboard=storyboard)
            env_fn = lambda: normalize_firewall_collections(make_custom_env(
                scenario_path,
                max_episode_steps=MAX_STEPS,
                llm_guidance=True,
                subgoal_manager=subgoal_manager,
                intrinsic_reward=False,
                intrinsic_reward_lambda=10))
        elif agent_type == AGENT_TYPE_PPO:
            env_fn = lambda: normalize_firewall_collections(make_custom_env(
                scenario_path,
                max_episode_steps=MAX_STEPS))
    else:
        if agent_type == AGENT_TYPE_LGRL:
            subgoal_manager = OracleSubgoalManager(utils=utils, storyboard=storyboard)
            env_fn = make_env_reseedable(
                scenario_name,
                train_seeds,
                max_episode_steps=MAX_STEPS,
                llm_guidance=True,
                subgoal_manager=subgoal_manager,
                intrinsic_reward=False,
                intrinsic_reward_lambda=10
            )
        elif agent_type == AGENT_TYPE_PPO:
            env_fn = make_env_reseedable(
                scenario_name,
                train_seeds,
                max_episode_steps=MAX_STEPS
            )

    if model_path is not None:
        model = MaskablePPO.load(model_path)
    else:
        vec_env = DummyVecEnv([env_fn])
        vec_env = VecFrameStack(vec_env, n_stack=FRAME_MEMORY)

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

        ideal_step = IDEAL_STEPS.get(scenario_name)
        convergence_cb = None
        if ideal_step is not None:
            convergence_cb = ConvergenceCallback(ideal_steps=ideal_step, window_size=WINDOW_SIZE, margin=MARGIN)
        else:
            print(f"* WARNING: No IDEAL_STEPS configured for scenario '{scenario_name}'. Convergence metric disabled.")

        print("=================STARTING TRAINING=================")
        if convergence_cb is not None:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=convergence_cb)
        else:
            model.learn(total_timesteps=TOTAL_TIMESTEPS)

        # Save Model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        scenario_name_for_model = scenario_name if scenario_name is not None else scenario_path.split("/")[-1].split(".")[0]
        model_name = f"{agent_type}_{scenario_name_for_model}_({MAX_STEPS}_{TOTAL_EPISODES}ep)_{timestamp}"
        model.save(f"models/{model_name}")

        # Store CS metric for evaluation summary
        if convergence_cb is not None:
            convergence_speed = convergence_cb.convergence_timestep
        print("=================TRAINING COMPLETE=================")

    # ===== EVALUATION PHASE =====
    print("\n---------------------------------------")
    print("Evaluation phase:")
    print("---------------------------------------")

    eval_seeds = [1000, 1001, 1002, 1003]

    for seed in eval_seeds:
        if scenario_path is not None:
            if agent_type == AGENT_TYPE_LGRL:
                eval_env_fn = lambda: normalize_firewall_collections(make_custom_env(
                    scenario_path,
                    max_episode_steps=MAX_STEPS,
                    llm_guidance=True,
                    subgoal_manager=subgoal_manager,
                    intrinsic_reward=False,
                    intrinsic_reward_lambda=10))
            else:
                eval_env_fn = lambda: normalize_firewall_collections(make_custom_env(
                    scenario_path,
                    max_episode_steps=MAX_STEPS))
        else:
            if agent_type == AGENT_TYPE_LGRL:
                eval_env_fn = make_eval_env(
                    scenario_name,
                    seed,
                    max_episode_steps=MAX_STEPS,
                    llm_guidance=True,
                    subgoal_manager=subgoal_manager,
                    intrinsic_reward=False,
                    intrinsic_reward_lambda=10,
                )
            else:
                eval_env_fn = make_eval_env(scenario_name, seed, max_episode_steps=MAX_STEPS)

        eval_env = DummyVecEnv([eval_env_fn])
        eval_vec_env = VecFrameStack(eval_env, n_stack=FRAME_MEMORY)

        success_count = 0
        total_tokens_used = 0
        total_cumulative_reward = 0

        for ep in range(EVAL_EPISODES):
            print(f"\n=== Evaluation Episode {ep + 1}/{EVAL_EPISODES} ===")

            obs = eval_vec_env.reset()

            terminated = False
            truncated = False
            ep_steps = 0
            ep_reward = 0
            ep_token_usage = 0

            while not terminated and not truncated and ep_steps < MAX_STEPS:
                action_masks = np.array([custom_mask_fn(eval_vec_env.envs[0])])
                
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

                base_env = eval_vec_env.envs[0].env.unwrapped 
                action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                action_obj = base_env.action_space.get_action(action_idx)
                print(f"- Step {ep_steps + 1}: {action_obj}")
                
                obs, reward, done, info = eval_vec_env.step(action)
                terminated = done[0]
                truncated = info[0].get('TimeLimit.truncated', False)
                ep_steps += 1
                ep_reward += reward[0]

                if RENDER_OBS_STATE:
                    base_env.render()
                    base_env.render_state()

                print(f"  Reward: {reward[0]:.2f}, Cumulative: {ep_reward:.2f}")

            done = bool(terminated and not truncated)
            success_count += 1 if done else 0
            total_cumulative_reward += ep_reward
            total_tokens_used += ep_token_usage

            print(f"\nEvaluation {ep + 1}/{EVAL_EPISODES} complete:")
            print(f"  - Steps: {ep_steps}")
            print(f"  - Total reward: {ep_reward:.2f}")
            print(f"  - Goal reached: {done}")
            print(f"  - Truncated: {truncated}")
    
    # Final Metrics Calculation
    success_rate = (success_count / EVAL_EPISODES) * 100
    avg_cumulative_reward = total_cumulative_reward / EVAL_EPISODES
    avg_tokens_used = total_tokens_used / EVAL_EPISODES

    print("\n=======================================")
    print("Evaluation Summary:")
    print("=======================================")
    print(f"Agent Type                 : {agent_type.upper()}")
    print(f"Total Evaluation Episodes  : {EVAL_EPISODES}")
    print(f"Convergence Speed          : {convergence_speed} timesteps") 
    print(f"Success Rate               : {success_rate:.2f}%")
    print(f"Average Cumulative Reward  : {avg_cumulative_reward:.2f}")
    print(f"Average Tokens Used        : {avg_tokens_used:.2f}")

    return done, truncated, ep_steps

# Print usage information
def usage():
    print("\nOVERVIEW: Run demo of the PenGym training framework for pentesting agents\n")
    print("USAGE: python3 run.py [options] <CONFIG_FILE> \n")
    print("OPTIONS:")
    print("-h, --help                     Display this help message and exit")
    print("-a, --agent_type <AGENT_TYPE>  Agent type (ppo/lgrl)")
    print("-l, --load_model <MODEL_PATH>  Load a pre-trained model from the specified path")
    print("-d, --disable_pengym           Disable PenGym execution in cyber range")
    print("-n, --nasim_simulation         Enable NASim simulation execution")

#############################################################################
# Main program
#############################################################################
def main(args):

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='* %(levelname)s: %(filename)s: %(message)s')


    print("#########################################################################")
    print("PenGym: Pentesting Training Framework for Reinforcement Learning Agents")
    print("#########################################################################")

    # Default argument values
    agent_type = DEFAULT_AGENT_TYPE
    config_path = None
    model_path = None

    # Parse command line arguments
    try:
        # Make sure to add ':' for short-form and '=' for long-form options that require an argument
        opts, trailing_args = getopt.getopt(args, "ha:l:dn",
                                            ["help", "agent_type=", "load_model=", "disable_pengym", "nasim_simulation"])
    except getopt.GetoptError as err:
        logging.error(f"Command-line argument error: {str(err)}")
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-a", "--agent_type"):
            agent_type = arg
        elif opt in ("-l", "--load_model"):
            model_path = arg
        elif opt in ("-d", "--disable_pengym"):
            utils.ENABLE_PENGYM = False
        elif opt in ("-n", "--nasim_simulation"):
            utils.ENABLE_NASIM = True
        else:
            # Nothing to do, since unrecognized options are caught by
            # the getopt.GetoptError exception above
            pass

    # Get path of configuration file
    try:
        config_path = trailing_args[0]
    except Exception as e:
        logging.error(f"Configuration file is not specified")
        usage()
        sys.exit(2)
    
    if model_path is not None:
        filename = model_path + ".zip"

        if isfile(filename):
            print(f"* Load pre-trained model from '{filename}'...")
        else:
            logging.error("Specified model path does not exist")
            sys.exit(2)

    # Print parameters
    print(f"* Execution parameters:")
    print(f"  - Agent type: {agent_type}")
    print(f"  - Load pre-trained model: {model_path is not None}")
    print(f"  - PenGym cyber range execution enabled: {utils.ENABLE_PENGYM}")
    print(f"  - NASim simulation execution enabled: {utils.ENABLE_NASIM}")

    # Check execution parameters
    if not (utils.ENABLE_PENGYM or utils.ENABLE_NASIM):
        logging.error("Either PenGym or NASim must be enabled")
        usage()
        sys.exit(2)

    print(f"* Read configuration from '{config_path}'...")
    utils.init_config_info(config_path)

    # Create an experiment environment using scenario path
    scenario_path = utils.replace_file_path(utils.config_info, storyboard.SCENARIO_FILE)
    scenario_name = utils.config_info.get(storyboard.SCENARIO_NAME)

    generation_enabled = False

    gen_scenario = ["tiny-gen", "tiny-gen-rangoal", "small-gen", "small-gen-rangoal", "medium-gen", "large-gen", "huge-gen", "pocp-1-gen", "pocp-2-gen"]

    if (scenario_name in gen_scenario):
        generation_enabled = True
    
    print(f"  - Scenario generation enabled: {generation_enabled}")

    if generation_enabled:
        print(f"* Generated scenario configuration: '{scenario_name}'")

    if utils.ENABLE_PENGYM:
        print(f"* Read configuration from '{config_path}'...")
        utils.init_config_info(config_path)
        
        print("* Initialize MSF RPC client...")
        utils.init_msfrpc_client()
        
        print("* Initialize Nmap Scanner...")
        utils.init_nmap_scanner()
        
        # Create host map dictionary
        range_detail_file = utils.replace_file_path(database=utils.config_info,
                                                     file_name=storyboard.RANGE_DETAILS_FILE)

        utils.init_host_map(range_details_file=range_detail_file)

        # Initializer map of service ports
        utils.init_service_port_map()
    
        # Deactivate bridge that not conneccted to Internet
        utils.init_bridge_setup(range_details_file=range_detail_file)

    # Run experiment using a PPO agent
    if agent_type == AGENT_TYPE_PPO:
        print("* Perform pentesting using a PPO agent...")
        if model_path is not None:
            print(f"* Load pre-trained model from '{model_path}.zip'...")
            if generation_enabled:
                done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_name=scenario_name, model_path=model_path)
            else:
                done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_path, model_path=model_path)
        else:
            print("* Training from scratch...")
            if generation_enabled:
                done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_name=scenario_name)
            else:
                done, truncated, step_count = run_agent(AGENT_TYPE_PPO, scenario_path)
    
    # Run experiment using a LGRL agent
    elif agent_type == AGENT_TYPE_LGRL:
        print("* Perform pentesting using a LGRL agent...")
        if generation_enabled:
            done, truncated, step_count = run_agent(AGENT_TYPE_LGRL, scenario_name=scenario_name)
        else:
            done, truncated, step_count = run_agent(AGENT_TYPE_LGRL, scenario_path)

    else:
        logging.error(f"Unrecognized agent type: '{agent_type}'")
        usage()
        sys.exit(1)

    # Print execution status
    if done:
        # All the goals in the scenario file were reached
        print(f"* NORMAL execution: {step_count} steps")
    elif truncated:
        # Execution was truncated before reaching all the goals (for random agents, etc.)
        print(f"* TRUNCATED execution: {step_count} steps")
    else:
        # Execution finished before reaching all the goals (for deterministic agents)
        print(f"* INCOMPLETE execution: {step_count} steps")

    if utils.ENABLE_PENGYM:
        print("* Clean up MSF RPC client...")
        utils.cleanup_msfrpc_client()

        print("* Restore the to intial state of the firewalls for all hosts...")
        utils.save_restore_firewall_rules_all_hosts(flag=storyboard.RESTORE)

#############################################################################
# Run program
if __name__ == "__main__":
    start = time.time()
    main(sys.argv[1:])
    end = time.time()
    print(f"Execution Time: {end-start:1.6f}s")